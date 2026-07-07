use anyhow::Result;
use ash::vk;

use crate::gfx::base::VulkanBase;

/// The shared "subject view" render target. The subject scene is drawn directly
/// into this image by Skia (see `skia_offscreen.rs`), which shares this app's
/// `ash` Vulkan device. The subject window's swapchain image is a blit of it,
/// and the operator window's imgui UI samples it directly, so both views always
/// show the same content.
pub struct OffscreenTarget {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
}

impl OffscreenTarget {
    pub const FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

    pub fn new(base: &VulkanBase, extent: vk::Extent2D) -> Result<Self> {
        let image = unsafe {
            base.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(Self::FORMAT)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                            | vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?
        };

        let requirements = unsafe { base.device.get_image_memory_requirements(image) };
        let memory_type_index = base
            .find_memory_type_index(requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let memory = unsafe {
            base.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type_index),
                None,
            )?
        };
        unsafe { base.device.bind_image_memory(image, memory, 0)? };

        let view = unsafe {
            base.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(Self::FORMAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )?
        };

        let sampler = unsafe {
            base.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(1.0),
                None,
            )?
        };

        Ok(Self {
            image,
            memory,
            view,
            sampler,
            extent,
            format: Self::FORMAT,
        })
    }

    /// Records a pipeline barrier transitioning the whole image between layouts.
    pub fn transition(
        &self,
        base: &VulkanBase,
        command_buffer: vk::CommandBuffer,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src: (vk::PipelineStageFlags, vk::AccessFlags),
        dst: (vk::PipelineStageFlags, vk::AccessFlags),
    ) {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(src.1)
            .dst_access_mask(dst.1);
        unsafe {
            base.device.cmd_pipeline_barrier(
                command_buffer,
                src.0,
                dst.0,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    pub fn destroy(self, base: &VulkanBase) {
        unsafe {
            base.device.destroy_sampler(self.sampler, None);
            base.device.destroy_image_view(self.view, None);
            base.device.destroy_image(self.image, None);
            base.device.free_memory(self.memory, None);
        }
    }
}
