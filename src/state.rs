use crate::pb::thalamus_grpc::observable_change::Action;
use anyhow::{bail, Context, Result};
use serde_json::Value;

/// A single step in a bracket-notation observable address, e.g. `['nodes'][2]['name']`.
#[derive(Debug, PartialEq, Eq)]
enum PathToken {
    Key(String),
    Index(usize),
}

/// Parses addresses of the form `['a']['b'][3]` as sent by Thalamus over
/// `observable_bridge_v2`. An empty (or whitespace-only) address refers to the root value.
fn parse_address(address: &str) -> Result<Vec<PathToken>> {
    let chars: Vec<char> = address.trim().chars().collect();
    let mut tokens = Vec::new();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        if chars[i] != '[' {
            bail!("expected '[' at position {i} in address {address:?}");
        }
        i += 1;
        if i >= n {
            bail!("unterminated address {address:?}");
        }

        if chars[i] == '\'' || chars[i] == '"' {
            let quote = chars[i];
            i += 1;
            let mut key = String::new();
            while i < n && chars[i] != quote {
                if chars[i] == '\\' && i + 1 < n {
                    key.push(chars[i + 1]);
                    i += 2;
                } else {
                    key.push(chars[i]);
                    i += 1;
                }
            }
            if i >= n {
                bail!("unterminated quoted key in address {address:?}");
            }
            i += 1; // closing quote
            tokens.push(PathToken::Key(key));
        } else {
            let start = i;
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i == start {
                bail!("expected index or quoted key in address {address:?}");
            }
            let idx: usize = chars[start..i]
                .iter()
                .collect::<String>()
                .parse()
                .with_context(|| format!("invalid index in address {address:?}"))?;
            tokens.push(PathToken::Index(idx));
        }

        if i >= n || chars[i] != ']' {
            bail!("expected ']' in address {address:?}");
        }
        i += 1;
        while i < n && chars[i].is_whitespace() {
            i += 1;
        }
    }

    Ok(tokens)
}

/// Thalamus serializes string values via a raw (unquoted) string rather than
/// a JSON string literal, so plain `serde_json::from_str` fails for them.
/// Fall back to treating the raw text as a JSON string in that case.
fn parse_value(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Steps into (creating containers/slots as needed) the given token, returning
/// a mutable reference to that slot.
fn step_into<'a>(current: &'a mut Value, token: &PathToken) -> &'a mut Value {
    match token {
        PathToken::Key(key) => {
            if !current.is_object() {
                *current = Value::Object(Default::default());
            }
            current
                .as_object_mut()
                .unwrap()
                .entry(key.clone())
                .or_insert(Value::Null)
        }
        PathToken::Index(idx) => {
            if !current.is_array() {
                *current = Value::Array(Vec::new());
            }
            let arr = current.as_array_mut().unwrap();
            while arr.len() <= *idx {
                arr.push(Value::Null);
            }
            &mut arr[*idx]
        }
    }
}

/// Applies one `ObservableChange` (address/value/action) to the JSON tree
/// representing application state, mirroring the semantics of Thalamus's own
/// C#/Python `observable_bridge_v2` clients.
pub fn apply_change(root: &mut Value, address: &str, value: &str, action: Action) -> Result<()> {
    let tokens = parse_address(address)?;

    if tokens.is_empty() {
        *root = match action {
            Action::Set => parse_value(value),
            Action::Delete => Value::Null,
        };
        return Ok(());
    }

    let mut current = root;
    for token in &tokens[..tokens.len() - 1] {
        current = step_into(current, token);
    }
    let last = tokens.last().unwrap();

    match action {
        Action::Set => {
            *step_into(current, last) = parse_value(value);
        }
        Action::Delete => match last {
            PathToken::Key(key) => {
                if let Value::Object(map) = current {
                    map.remove(key);
                }
            }
            PathToken::Index(idx) => {
                if let Value::Array(arr) = current {
                    if *idx < arr.len() {
                        arr.remove(*idx);
                    }
                }
            }
        },
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn root_replace() {
        let mut state = json!(null);
        apply_change(&mut state, "", r#"{"a":1}"#, Action::Set).unwrap();
        assert_eq!(state, json!({"a": 1}));
    }

    #[test]
    fn nested_set_creates_path() {
        let mut state = json!({});
        apply_change(&mut state, "['nodes'][2]['name']", "\"probe\"", Action::Set).unwrap();
        assert_eq!(state["nodes"][2]["name"], json!("probe"));
        assert_eq!(state["nodes"][0], Value::Null);
    }

    #[test]
    fn raw_unquoted_string_value_falls_back() {
        let mut state = json!({});
        apply_change(&mut state, "['label']", "hello world", Action::Set).unwrap();
        assert_eq!(state["label"], json!("hello world"));
    }

    #[test]
    fn delete_removes_key() {
        let mut state = json!({"a": 1, "b": 2});
        apply_change(&mut state, "['a']", "", Action::Delete).unwrap();
        assert_eq!(state, json!({"b": 2}));
    }

    #[test]
    fn delete_removes_index() {
        let mut state = json!({"list": [1, 2, 3]});
        apply_change(&mut state, "['list'][1]", "", Action::Delete).unwrap();
        assert_eq!(state["list"], json!([1, 3]));
    }
}
