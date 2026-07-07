use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use raw_window_handle::{HandleError, HasDisplayHandle, HasWindowHandle};
use std::ffi::CString;

/// `raw_window_handle::HandleError` doesn't implement `std::error::Error`, so it
/// can't cross an anyhow `?` boundary directly.
fn handle_err(e: HandleError) -> anyhow::Error {
    anyhow!("{e:?}")
}

/// Vulkan objects shared by every window: one instance, one physical device, one
/// logical device and a single graphics/present queue used for all rendering.
pub struct VulkanBase {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface_loader: ash::khr::surface::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub device: ash::Device,
    pub queue: vk::Queue,
}

impl VulkanBase {
    /// Creates the shared Vulkan instance/device, picking a physical device and
    /// queue family that can present to every surface in `surfaces`.
    pub fn new(windows: &[&winit::window::Window]) -> Result<(Self, Vec<vk::SurfaceKHR>)> {
        let entry = unsafe { ash::Entry::load().context("loading the Vulkan loader")? };

        let mut required_extensions: Vec<*const std::os::raw::c_char> = Vec::new();
        for window in windows {
            let display_handle = window.display_handle().map_err(handle_err)?.as_raw();
            for ext in ash_window::enumerate_required_extensions(display_handle)? {
                if !required_extensions.contains(ext) {
                    required_extensions.push(*ext);
                }
            }
        }

        let app_name = CString::new("rust-executor")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_1);

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&required_extensions);

        let instance = unsafe { entry.create_instance(&instance_create_info, None)? };
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        let surfaces = windows
            .iter()
            .map(|window| unsafe {
                let display_handle = window.display_handle().map_err(handle_err)?.as_raw();
                let window_handle = window.window_handle().map_err(handle_err)?.as_raw();
                ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
                    .context("creating window surface")
            })
            .collect::<Result<Vec<_>>>()?;

        let (physical_device, queue_family_index) =
            unsafe { Self::pick_physical_device(&instance, &surface_loader, &surfaces)? };

        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);
        let queue_create_infos = [queue_create_info];

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok((
            Self {
                entry,
                instance,
                surface_loader,
                physical_device,
                queue_family_index,
                device,
                queue,
            },
            surfaces,
        ))
    }

    /// Finds the first physical device with a queue family that supports
    /// graphics and can present to every given surface.
    unsafe fn pick_physical_device(
        instance: &ash::Instance,
        surface_loader: &ash::khr::surface::Instance,
        surfaces: &[vk::SurfaceKHR],
    ) -> Result<(vk::PhysicalDevice, u32)> {
        for physical_device in unsafe { instance.enumerate_physical_devices()? } {
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            for (index, family) in queue_families.iter().enumerate() {
                let index = index as u32;
                if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    continue;
                }
                let supports_all_surfaces = surfaces.iter().all(|surface| unsafe {
                    surface_loader
                        .get_physical_device_surface_support(physical_device, index, *surface)
                        .unwrap_or(false)
                });
                if supports_all_surfaces {
                    return Ok((physical_device, index));
                }
            }
        }
        bail!("no Vulkan physical device with a queue family that can present to all windows")
    }

    /// Finds a memory type index satisfying `type_bits` and `required_properties`.
    pub fn find_memory_type_index(
        &self,
        type_bits: u32,
        required_properties: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        let mem_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        for i in 0..mem_properties.memory_type_count {
            let suitable = (type_bits & (1 << i)) != 0;
            let has_properties = mem_properties.memory_types[i as usize]
                .property_flags
                .contains(required_properties);
            if suitable && has_properties {
                return Ok(i);
            }
        }
        bail!("no suitable Vulkan memory type found")
    }
}
