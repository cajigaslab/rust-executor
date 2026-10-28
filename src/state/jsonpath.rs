//! Port of Thalamus' JSONPath support (`jsonpath_grammar.hpp` and
//! `get_jsonpath`/`set_jsonpath`/`delete_jsonpath` in `state.cpp`), the
//! address language `observable_bridge_v2` changes are written in.
//!
//! Grammar (whitespace is ignored between tokens):
//!
//! ```text
//! query      := '$'? name? segment*
//! segment    := '[' selector ']' | '.' name
//! selector   := string | int | '?' filter | '?' '(' filter ')'
//! filter     := and ('||' and)*
//! and        := comparison ('&&' comparison)*
//! comparison := operand ('==' | '!=' | '<=' | '>=' | '<' | '>') operand
//! operand    := literal | ('@' | '$') (('[' (string | int) ']') | '.' name)*
//! literal    := number | string | 'true' | 'false' | 'null'
//! ```
//!
//! A filter selects the first child (dict value or list item) it matches,
//! as in C++. Beyond the C++ grammar this accepts a leading bare member
//! name (`nodes[0]`, as Python's `jsonpath_ng` does), parentheses around a
//! filter, floats, backslash escapes in strings (matching how Python
//! `repr`s keys into addresses), negative list indices counting from the
//! end, and `<`/`>` (which C++ parses but never matches).

use std::cmp::Ordering;

use anyhow::{Result, anyhow, bail};
use serde_json::Value as Json;

use super::{State, StateKey, StateValue};

#[derive(Debug, Clone, PartialEq)]
enum Segment {
  Name(String),
  Index(i64),
  /// OR of ANDs of comparisons.
  Filter(Vec<Vec<Comparison>>),
}

#[derive(Debug, Clone, PartialEq)]
struct Comparison {
  lhs: Operand,
  op: Op,
  rhs: Operand,
}

#[derive(Debug, Clone, PartialEq)]
enum Operand {
  Literal(Json),
  /// `@...` (relative to the candidate child) or `$...` (absolute); only
  /// `Name`/`Index` segments.
  Query { absolute: bool, segments: Vec<Segment> },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Op {
  Eq,
  Ne,
  Lt,
  Le,
  Gt,
  Ge,
}

fn is_name_first(c: char) -> bool {
  c.is_ascii_alphabetic() || c == '_'
}

fn is_name_char(c: char) -> bool {
  c.is_ascii_alphanumeric() || c == '_'
}

struct Parser<'a> {
  text: &'a str,
  chars: Vec<char>,
  pos: usize,
}

impl<'a> Parser<'a> {
  fn new(text: &'a str) -> Self {
    Self {
      text,
      chars: text.chars().collect(),
      pos: 0,
    }
  }

  fn error(&self, what: &str) -> anyhow::Error {
    anyhow!("{what} at position {} in JSONPath {:?}", self.pos, self.text)
  }

  fn skip_ws(&mut self) {
    while self.chars.get(self.pos).is_some_and(|c| c.is_whitespace()) {
      self.pos += 1;
    }
  }

  fn peek(&mut self) -> Option<char> {
    self.skip_ws();
    self.chars.get(self.pos).copied()
  }

  fn eat(&mut self, token: &str) -> bool {
    self.skip_ws();
    let token: Vec<char> = token.chars().collect();
    if self.chars[self.pos..].starts_with(&token) {
      self.pos += token.len();
      true
    } else {
      false
    }
  }

  fn expect(&mut self, token: &str) -> Result<()> {
    if self.eat(token) {
      Ok(())
    } else {
      Err(self.error(&format!("expected {token:?}")))
    }
  }

  fn query(&mut self) -> Result<Vec<Segment>> {
    self.eat("$");
    let mut segments = Vec::new();
    if self.peek().is_some_and(is_name_first) {
      segments.push(Segment::Name(self.name()?));
    }
    loop {
      if self.eat("[") {
        segments.push(self.selector()?);
        self.expect("]")?;
      } else if self.eat(".") {
        segments.push(Segment::Name(self.name()?));
      } else {
        break;
      }
    }
    if self.peek().is_some() {
      return Err(self.error("unexpected character"));
    }
    Ok(segments)
  }

  fn selector(&mut self) -> Result<Segment> {
    match self.peek() {
      Some('\'' | '"') => Ok(Segment::Name(self.string()?)),
      Some('?') => {
        self.pos += 1;
        let parenthesized = self.eat("(");
        let filter = self.filter()?;
        if parenthesized {
          self.expect(")")?;
        }
        Ok(Segment::Filter(filter))
      }
      Some(c) if c == '-' || c.is_ascii_digit() => Ok(Segment::Index(self.int()?)),
      _ => Err(self.error("expected a quoted name, an index or a filter")),
    }
  }

  fn name(&mut self) -> Result<String> {
    self.skip_ws();
    if !self.chars.get(self.pos).copied().is_some_and(is_name_first) {
      return Err(self.error("expected a member name"));
    }
    let start = self.pos;
    while self.chars.get(self.pos).copied().is_some_and(is_name_char) {
      self.pos += 1;
    }
    Ok(self.chars[start..self.pos].iter().collect())
  }

  fn string(&mut self) -> Result<String> {
    self.skip_ws();
    let quote = self.chars[self.pos];
    self.pos += 1;
    let mut out = String::new();
    loop {
      match self.chars.get(self.pos).copied() {
        None => return Err(self.error("unterminated string")),
        Some(c) if c == quote => {
          self.pos += 1;
          return Ok(out);
        }
        Some('\\') => {
          let Some(&escaped) = self.chars.get(self.pos + 1) else {
            return Err(self.error("unterminated string"));
          };
          out.push(escaped);
          self.pos += 2;
        }
        Some(c) => {
          out.push(c);
          self.pos += 1;
        }
      }
    }
  }

  fn number(&mut self) -> Result<Json> {
    self.skip_ws();
    let start = self.pos;
    if self.chars.get(self.pos) == Some(&'-') {
      self.pos += 1;
    }
    let digits = |p: &mut Self| {
      let digits_start = p.pos;
      while p.chars.get(p.pos).is_some_and(|c| c.is_ascii_digit()) {
        p.pos += 1;
      }
      p.pos > digits_start
    };
    if !digits(self) {
      return Err(self.error("expected a number"));
    }
    let mut is_float = false;
    if self.chars.get(self.pos) == Some(&'.') && self.chars.get(self.pos + 1).is_some_and(|c| c.is_ascii_digit()) {
      self.pos += 1;
      digits(self);
      is_float = true;
    }
    let text: String = self.chars[start..self.pos].iter().collect();
    if is_float {
      let value: f64 = text.parse().map_err(|_| self.error("invalid number"))?;
      Ok(Json::from(value))
    } else {
      let value: i64 = text.parse().map_err(|_| self.error("invalid integer"))?;
      Ok(Json::from(value))
    }
  }

  fn int(&mut self) -> Result<i64> {
    self.number()?.as_i64().ok_or_else(|| self.error("expected an integer"))
  }

  fn filter(&mut self) -> Result<Vec<Vec<Comparison>>> {
    let mut alternatives = vec![self.conjunction()?];
    while self.eat("||") {
      alternatives.push(self.conjunction()?);
    }
    Ok(alternatives)
  }

  fn conjunction(&mut self) -> Result<Vec<Comparison>> {
    let mut comparisons = vec![self.comparison()?];
    while self.eat("&&") {
      comparisons.push(self.comparison()?);
    }
    Ok(comparisons)
  }

  fn comparison(&mut self) -> Result<Comparison> {
    let lhs = self.operand()?;
    let op = self.op()?;
    let rhs = self.operand()?;
    Ok(Comparison { lhs, op, rhs })
  }

  fn op(&mut self) -> Result<Op> {
    // Two-character operators first so `<=` isn't read as `<`.
    for (token, op) in [
      ("==", Op::Eq),
      ("!=", Op::Ne),
      ("<=", Op::Le),
      (">=", Op::Ge),
      ("<", Op::Lt),
      (">", Op::Gt),
    ] {
      if self.eat(token) {
        return Ok(op);
      }
    }
    Err(self.error("expected a comparison operator"))
  }

  fn operand(&mut self) -> Result<Operand> {
    match self.peek() {
      Some(c @ ('@' | '$')) => {
        self.pos += 1;
        let mut segments = Vec::new();
        loop {
          if self.peek() == Some('[') {
            self.pos += 1;
            match self.peek() {
              Some('\'' | '"') => segments.push(Segment::Name(self.string()?)),
              Some(c) if c == '-' || c.is_ascii_digit() => segments.push(Segment::Index(self.int()?)),
              _ => return Err(self.error("expected a quoted name or an index")),
            }
            self.expect("]")?;
          } else if self.peek() == Some('.') && self.chars.get(self.pos + 1).copied().is_some_and(is_name_first) {
            self.pos += 1;
            segments.push(Segment::Name(self.name()?));
          } else {
            break;
          }
        }
        Ok(Operand::Query {
          absolute: c == '$',
          segments,
        })
      }
      Some('\'' | '"') => Ok(Operand::Literal(Json::String(self.string()?))),
      Some(c) if c == '-' || c.is_ascii_digit() => Ok(Operand::Literal(self.number()?)),
      _ => {
        for (keyword, value) in [("true", Json::Bool(true)), ("false", Json::Bool(false)), ("null", Json::Null)] {
          if self.eat(keyword) {
            return Ok(Operand::Literal(value));
          }
        }
        Err(self.error("expected a literal, '@' or '$'"))
      }
    }
  }
}

fn parse(query: &str) -> Result<Vec<Segment>> {
  Parser::new(query).query()
}

/// The child of `current` that `segment` selects, with its key.
fn select(current: &StateValue, segment: &Segment, root: &StateValue) -> Option<(StateKey, StateValue)> {
  let state = current.as_state()?;
  match segment {
    Segment::Name(name) => {
      let key = StateKey::String(name.clone());
      state.get(key.clone()).map(|value| (key, value))
    }
    Segment::Index(index) => {
      if state.is_list() {
        let index = if *index < 0 { state.len() as i64 + index } else { *index };
        let key = StateKey::Int(index);
        state.get(key.clone()).map(|value| (key, value))
      } else {
        // C++ dicts can hold integer keys; JSON-sourced ones only have strings.
        [StateKey::Int(*index), StateKey::String(index.to_string())]
          .into_iter()
          .find_map(|key| state.get(key.clone()).map(|value| (key, value)))
      }
    }
    Segment::Filter(filter) => state
      .entries()
      .into_iter()
      .find(|(_, child)| matches_filter(filter, child, root)),
  }
}

/// Follows `segments` from `start`; `Null` if any step finds nothing.
fn resolve(root: &StateValue, start: &StateValue, segments: &[Segment]) -> StateValue {
  let mut current = start.clone();
  for segment in segments {
    match select(&current, segment, root) {
      Some((_, value)) => current = value,
      None => return StateValue::Null,
    }
  }
  current
}

fn matches_filter(filter: &[Vec<Comparison>], current: &StateValue, root: &StateValue) -> bool {
  filter
    .iter()
    .any(|conjunction| conjunction.iter().all(|comparison| compare(comparison, current, root)))
}

fn operand_value(operand: &Operand, current: &StateValue, root: &StateValue) -> StateValue {
  match operand {
    Operand::Literal(json) => StateValue::from_json(json),
    Operand::Query { absolute, segments } => resolve(root, if *absolute { root } else { current }, segments),
  }
}

fn as_number(value: &StateValue) -> Option<f64> {
  match value {
    StateValue::Int(i) => Some(*i as f64),
    StateValue::Float(f) => Some(*f),
    _ => None,
  }
}

fn order(lhs: &StateValue, rhs: &StateValue) -> Option<Ordering> {
  match (lhs, rhs) {
    (StateValue::String(l), StateValue::String(r)) => Some(l.cmp(r)),
    _ => as_number(lhs)?.partial_cmp(&as_number(rhs)?),
  }
}

fn compare(comparison: &Comparison, current: &StateValue, root: &StateValue) -> bool {
  let lhs = operand_value(&comparison.lhs, current, root);
  let rhs = operand_value(&comparison.rhs, current, root);
  // Ints and floats compare numerically (`1 == 1.0`).
  let equal = match (as_number(&lhs), as_number(&rhs)) {
    (Some(l), Some(r)) => l == r,
    _ => lhs == rhs,
  };
  match comparison.op {
    Op::Eq => equal,
    Op::Ne => !equal,
    Op::Lt => order(&lhs, &rhs).is_some_and(Ordering::is_lt),
    Op::Le => order(&lhs, &rhs).is_some_and(Ordering::is_le),
    Op::Gt => order(&lhs, &rhs).is_some_and(Ordering::is_gt),
    Op::Ge => order(&lhs, &rhs).is_some_and(Ordering::is_ge),
  }
}

/// The value at `query` under `root`, or `Null` if there's nothing there.
pub fn get_jsonpath(root: &State, query: &str) -> Result<StateValue> {
  let segments = parse(query)?;
  let root = root.as_value();
  Ok(resolve(&root, &root, &segments))
}

/// Sets `value` at `query` under `root`. If a collection of the same kind
/// is already there it's assigned into recursively (keeping it, and its
/// observers — Python's behavior), otherwise the value is replaced. Setting
/// a list index past the end pads it with nulls first (C++'s behavior). An
/// empty query assigns to `root` itself.
pub fn set_jsonpath(root: &State, query: &str, value: &Json, from_remote: bool) -> Result<()> {
  let segments = parse(query)?;
  let Some((last, parents)) = segments.split_last() else {
    return root.assign_json_with(value, None, from_remote);
  };

  let root_value = root.as_value();
  let parent_value = resolve(&root_value, &root_value, parents);
  let Some(parent) = parent_value.as_state() else {
    bail!("{query:?}: there's no collection to set into");
  };

  let existing = select(&parent_value, last, &root_value);
  if let Some((_, StateValue::Dict(target) | StateValue::List(target))) = &existing
    && (target.is_dict() && value.is_object() || target.is_list() && value.is_array())
  {
    return target.assign_json_with(value, None, from_remote);
  }

  let key = match (existing, last) {
    (Some((key, _)), _) => key,
    (None, Segment::Name(name)) => {
      if parent.is_list() {
        bail!("{query:?}: can't index a list by name");
      }
      StateKey::String(name.clone())
    }
    (None, Segment::Index(index)) => {
      if parent.is_list() {
        let Ok(index) = usize::try_from(*index) else {
          bail!("{query:?}: index out of range");
        };
        for _ in parent.len()..index {
          parent.push_with(StateValue::Null, None, from_remote);
        }
      }
      StateKey::Int(*index)
    }
    (None, Segment::Filter(_)) => bail!("{query:?}: the filter matched nothing"),
  };
  parent.set_with(key, StateValue::from_json(value), None, from_remote);
  Ok(())
}

/// Removes whatever is at `query` under `root`; nothing happens if there's
/// nothing there. The root itself can't be deleted.
pub fn delete_jsonpath(root: &State, query: &str, from_remote: bool) -> Result<()> {
  let segments = parse(query)?;
  let Some((last, parents)) = segments.split_last() else {
    bail!("can't delete the root");
  };
  let root_value = root.as_value();
  let parent_value = resolve(&root_value, &root_value, parents);
  let Some(parent) = parent_value.as_state() else {
    return Ok(());
  };
  if let Some((key, _)) = select(&parent_value, last, &root_value) {
    parent.remove_with(key, None, from_remote);
  }
  Ok(())
}

#[cfg(test)]
pub(super) fn parse_for_test(query: &str) -> Result<()> {
  parse(query).map(|_| ())
}
