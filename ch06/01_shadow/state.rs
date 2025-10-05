use std::sync::Arc;
use bytemuck::cast_slice;
use cgmath::{Matrix, Matrix4, SquareMatrix};
use rand::Rng;
use std::mem;
use winit::{
    event_loop::ActiveEventLoop,
    keyboard::KeyCode,
    window::Window,
};

use wgpu_lighting::common_instance;
use wgpu_lighting::wgpu_simplified as ws;

#[derive(Debug)]
struct Scene {
    translation: [f32; 3],
    rotation: [f32; 3],
    scale: [f32; 3],
    v: f32,
}

fn create_transform_mat() -> (Vec<Scene>, Vec<[f32; 16]>, Vec<[f32; 16]>, Vec<[f32; 4]>) {
    let mut scenes: Vec<Scene> = vec![];
    let mut model_mat: Vec<[f32; 16]> = vec![];
    let mut normal_mat: Vec<[f32; 16]> = vec![];
    let mut color_vec: Vec<[f32; 4]> = vec![];

    let mut rng = rand::rng();

    // add a cube as floor
    let translation_cube = [0.0f32, -13.0, -20.0];
    let rotation_cube = [0.0f32, 0.0, 0.0];
    let scale_cube = [30.0f32, 0.1, 20.0];
    let m_cube = ws::create_model_mat(translation_cube, rotation_cube, scale_cube);
    let n_cube = (m_cube.invert().unwrap()).transpose();
    let c_cube = [0.5f32, 0.5, 0.7, 1.0];
    model_mat.push(*(m_cube.as_ref()));
    normal_mat.push(*(n_cube.as_ref()));
    color_vec.push(c_cube);
    scenes.push(Scene {
        translation: translation_cube,
        rotation: rotation_cube,
        scale: scale_cube,
        v: 0.0,
    });

    // add a torus
    let translation_torus = [0.0f32, -4.7, -20.0];
    let rotation_torus = [1.57f32, 0.0, 0.0];
    let scale_torus = [4.0f32, 4.0, 4.0];
    let m_torus = ws::create_model_mat(translation_torus, rotation_torus, scale_torus);
    let n_torus = (m_torus.invert().unwrap()).transpose();
    let c_torus = [rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>(), 1.0];
    model_mat.push(*(m_torus.as_ref()));
    normal_mat.push(*(n_torus.as_ref()));
    color_vec.push(c_torus);
    scenes.push(Scene {
        translation: translation_torus,
        rotation: rotation_torus,
        scale: scale_torus,
        v: 0.0,
    });

    // add spheres
    for _i in 2..20 {
        let mut v1 = -1.0f32;
        if rng.random::<f32>() > 0.5 {
            v1 = 1.0;
        }
        let tx = v1 * (4.0 + rng.random::<f32>() * 12.0);
        let translation_sphere = [tx, -11.0 + rng.random::<f32>() * 15.0, -20.0 + tx];
        let rotation_sphere = [rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>()];
        let s = [0.5f32, rng.random::<f32>()]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let scale_sphere = [s, s, s];
        let m_sphere = ws::create_model_mat(translation_sphere, rotation_sphere, scale_sphere);
        let n_sphere = (m_torus.invert().unwrap()).transpose();
        let c_sphere = [rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>(), 1.0];
        model_mat.push(*(m_sphere.as_ref()));
        normal_mat.push(*(n_sphere.as_ref()));
        color_vec.push(c_sphere);
        let v = v1
            * [0.09, rng.random::<f32>() / 10.0]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        scenes.push(Scene {
            translation: translation_sphere,
            rotation: rotation_sphere,
            scale: scale_sphere,
            v,
        });
    }

    (scenes, model_mat, normal_mat, color_vec)
}

pub struct State {
    init: ws::InitWgpu,
    pipelines: Vec<wgpu::RenderPipeline>,
    vertex_buffers: Vec<wgpu::Buffer>,
    index_buffers: Vec<wgpu::Buffer>,
    uniform_bind_groups: Vec<wgpu::BindGroup>,
    uniform_buffers: Vec<wgpu::Buffer>,
    view_mat: Matrix4<f32>,
    project_mat: Matrix4<f32>,
    depth_texture_views: Vec<wgpu::TextureView>,
    indices_lens: Vec<u32>,
    animation_speed: f32,

    light_position: [f32; 3],
    scenes: (Vec<Scene>, Vec<[f32; 16]>, Vec<[f32; 16]>, Vec<[f32; 4]>),

    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
}

impl State {
    pub async fn new(window: Arc<Window>, _sample_count: u32) -> Self {
        let init = ws::InitWgpu::init_wgpu(window, 1).await;

        let vs_shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("shadow_vert.wgsl"));
        let fs_shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("shadow_frag.wgsl"));
        let depth_shader = init
            .device
            .create_shader_module(wgpu::include_wgsl!("shadow_depth.wgsl"));

        // uniform data
        let camera_position = (0.0, 10.0, 20.0).into();
        let look_direction = (0.0, 0.0, 0.0).into();
        let up_direction = cgmath::Vector3::unit_y();

        let (view_mat, project_mat, vp_mat) = ws::create_vp_mat(
            camera_position,
            look_direction,
            up_direction,
            init.config.width as f32 / init.config.height as f32,
        );

        // create uniform view-projection buffers
        let vp_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("View-Projection Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        init.queue.write_buffer(
            &vp_uniform_buffer,
            0,
            cast_slice(vp_mat.as_ref() as &[f32; 16]),
        );

        // model storage buffer
        let objects_count = 20;
        let model_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Model Uniform Buffer"),
            size: 64 * objects_count as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // normal storage buffer
        let normal_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Normal Uniform Buffer"),
            size: 64 * objects_count as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // color storage buffer
        let color_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("color Uniform Buffer"),
            size: 16 * objects_count as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // uniform buffer for  light projection
        let light_projection_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Projection Uniform Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // light uniform buffer.
        let light_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Uniform Buffer"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        init.queue.write_buffer(
            &light_uniform_buffer,
            16,
            cast_slice(camera_position.as_ref() as &[f32; 3]),
        );

        // set specular light color to white
        let specular_color: [f32; 3] = [1.0, 1.0, 1.0];
        init.queue.write_buffer(
            &light_uniform_buffer,
            32,
            cast_slice(specular_color.as_ref()),
        );

        // material uniform buffer
        let material_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // set default material parameters
        let material = [0.4 as f32, 0.04, 0.4, 30.0];
        init.queue
            .write_buffer(&material_uniform_buffer, 0, cast_slice(material.as_ref()));

        // uniform bind group for vertex shader
        let (vert_bind_group_layout, vert_bind_group) = ws::create_bind_group_storage(
            &init.device,
            vec![
                wgpu::ShaderStages::VERTEX,
                wgpu::ShaderStages::VERTEX,
                wgpu::ShaderStages::VERTEX,
                wgpu::ShaderStages::VERTEX,
                wgpu::ShaderStages::VERTEX,
            ],
            vec![
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
            ],
            &[
                vp_uniform_buffer.as_entire_binding(),
                model_uniform_buffer.as_entire_binding(),
                normal_uniform_buffer.as_entire_binding(),
                light_projection_uniform_buffer.as_entire_binding(),
                color_uniform_buffer.as_entire_binding(),
            ],
        );

        // uniform bind group for shadow
        let (shadow_bind_group_layout, shadow_bind_group) = ws::create_bind_group_storage(
            &init.device,
            vec![wgpu::ShaderStages::VERTEX, wgpu::ShaderStages::VERTEX],
            vec![
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Uniform,
            ],
            &[
                model_uniform_buffer.as_entire_binding(),
                light_projection_uniform_buffer.as_entire_binding(),
            ],
        );
        // uniform bind group for fragment shader
        let (frag_bind_group_layout, frag_bind_group) = ws::create_bind_group(
            &init.device,
            vec![wgpu::ShaderStages::FRAGMENT, wgpu::ShaderStages::FRAGMENT],
            &[
                light_uniform_buffer.as_entire_binding(),
                material_uniform_buffer.as_entire_binding(),
            ],
        );

        // shadow texturebind group for fragment shader
        let shadow_depth_texture_view = ws::create_shadow_texture_view(&init, 2048, 2048);
        let shadow_depth_sampler = init.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            compare: Some(wgpu::CompareFunction::Less),
            ..Default::default()
        });
        let frag_bind_group_layout2 =
            init.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Depth,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                            count: None,
                        },
                    ],
                    label: Some("Fragment Bind Group Layout 2"),
                });

        let frag_bind_group2 = init.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &frag_bind_group_layout2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_depth_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shadow_depth_sampler),
                },
            ],
            label: Some("Fragment Bind  Group 2"),
        });

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<common_instance::Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };

        let pipeline_layout = init
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &vert_bind_group_layout,
                    &frag_bind_group_layout,
                    &frag_bind_group_layout2,
                ],
                push_constant_ranges: &[],
            });

        let mut ppl = ws::IRenderPipeline {
            vs_shader: Some(&vs_shader),
            fs_shader: Some(&fs_shader),
            pipeline_layout: Some(&pipeline_layout),
            vertex_buffer_layout: &[vertex_buffer_layout],
            ..Default::default()
        };
        let pipeline = ppl.new(&init);

        // pipeline for shadow
        let vertex_buffer_layout2 = wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<common_instance::Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };

        let pipeline_layout2 =
            init.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout 2"),
                    bind_group_layouts: &[&shadow_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline2 = init
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline 2"),
                layout: Some(&pipeline_layout2),
                vertex: wgpu::VertexState {
                    module: &depth_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_buffer_layout2],
                    compilation_options: Default::default(),
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let depth_texture_view = ws::create_depth_view(&init);

        // vertex and index buffers for objects:
        let (vertex_buffers, index_buffers, index_lens) =
            common_instance::create_object_buffers(&init.device);
        let (scenes, model_mat, normal_mat, color_vec) = create_transform_mat();
        init.queue
            .write_buffer(&color_uniform_buffer, 0, cast_slice(color_vec.as_ref()));

        Self {
            init,
            pipelines: vec![pipeline, pipeline2],
            vertex_buffers,
            index_buffers,
            uniform_bind_groups: vec![
                vert_bind_group,
                frag_bind_group,
                frag_bind_group2,
                shadow_bind_group,
            ],
            uniform_buffers: vec![
                vp_uniform_buffer,
                model_uniform_buffer,
                normal_uniform_buffer,
                light_projection_uniform_buffer,
                color_uniform_buffer,
                light_uniform_buffer,
                material_uniform_buffer,
            ],
            view_mat,
            project_mat,
            depth_texture_views: vec![depth_texture_view, shadow_depth_texture_view],
            indices_lens: index_lens,
            animation_speed: 1.0,

            light_position: [0.0, 100.0, 0.0],
            scenes: (scenes, model_mat, normal_mat, color_vec),

            ambient: material[0],
            diffuse: material[1],
            specular: material[2],
            shininess: material[3],
        }
    }

    pub fn window(&self) -> &Window {
        &self.init.window
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            // The surface needs to be reconfigured every time the window is resized.
            self.init.config.width = width;
            self.init.config.height = height;
            self.init
                .surface
                .configure(&self.init.device, &self.init.config);

            self.project_mat =
                ws::create_projection_mat(width as f32 / height as f32, true);
            self.depth_texture_views[0] = ws::create_depth_view(&self.init);
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        match (key, pressed) {
            (KeyCode::Escape, true) => {
                event_loop.exit();
            } 
            (KeyCode::KeyQ, _pressed) => {
                self.ambient += 0.01;
            }
            (KeyCode::KeyA, _pressed) => {
                self.ambient -= 0.05;
                if self.ambient < 0.0 {
                    self.ambient = 0.0;
                }
            }
            (KeyCode::KeyW, _pressed) => {
                self.diffuse += 0.05;
            }
            (KeyCode::KeyS, _pressed) => {
                self.diffuse -= 0.05;
                if self.diffuse < 0.0 {
                    self.diffuse = 0.0;
                }
            }
            (KeyCode::KeyE, _pressed) => {
                self.specular += 0.05;
            }
            (KeyCode::KeyD, _pressed) => {
                self.specular -= 0.05;
                if self.specular < 0.0 {
                    self.specular = 0.0;
                }
            }
            (KeyCode::KeyR, _pressed) => {
                self.shininess += 5.0;
            }
            (KeyCode::KeyF, _pressed) => {
                self.shininess -= 5.0;
                if self.shininess < 0.0 {
                    self.shininess = 0.0;
                }
            }
            _ => {},
        }
    }

    pub fn update(&mut self, dt: std::time::Duration) {
        // update uniform buffer
        let dt = self.animation_speed * dt.as_secs_f32();
        self.light_position[0] = 50.0 * dt.sin();
        self.light_position[2] = 50.0 * dt.cos();
        self.init.queue.write_buffer(
            &self.uniform_buffers[5],
            0,
            cast_slice(self.light_position.as_ref()),
        );

        let light_mat = ws::create_view_mat(
            self.light_position.into(),
            [0.0, 0.0, 0.0].into(),
            cgmath::Vector3::unit_y(),
        );
        let mut light_projection_mat = ws::create_ortho_mat(-40.0, 40.0, -40.0, 40.0, -50.0, 200.0);
        light_projection_mat = light_projection_mat * light_mat;
        self.init.queue.write_buffer(
            &self.uniform_buffers[3],
            0,
            cast_slice(light_projection_mat.as_ref() as &[f32; 16]),
        );

        // update positions
        let torus = &mut self.scenes.0[1];
        torus.rotation[1] = 2.0 * dt;
        let m_torus = ws::create_model_mat(torus.translation, torus.rotation, torus.scale);
        let n_torus = (m_torus.invert().unwrap()).transpose();
        self.scenes.1[1] = *m_torus.as_ref();
        self.scenes.2[1] = *n_torus.as_ref();

        for i in 2..20 {
            let sphere = &mut self.scenes.0[i];
            sphere.translation[1] += sphere.v;
            if sphere.translation[1] < -11.0 || sphere.translation[1] > 11.0 {
                sphere.v *= -1.0;
            }
            let m_sphere = ws::create_model_mat(sphere.translation, sphere.rotation, sphere.scale);
            let n_sphere = (m_sphere.invert().unwrap()).transpose();
            self.scenes.1[i] = *m_sphere.as_ref();
            self.scenes.2[i] = *n_sphere.as_ref();
        }
        self.init.queue.write_buffer(
            &self.uniform_buffers[1],
            0,
            cast_slice(self.scenes.1.as_ref()),
        );
        self.init.queue.write_buffer(
            &self.uniform_buffers[2],
            0,
            cast_slice(self.scenes.2.as_ref()),
        );

        let view_project_mat = self.project_mat * self.view_mat;
        let view_projection_ref: &[f32; 16] = view_project_mat.as_ref();
        self.init
            .queue
            .write_buffer(&self.uniform_buffers[0], 0, cast_slice(view_projection_ref));

        // update material
        let material = [self.ambient, self.diffuse, self.specular, self.shininess];
        self.init
            .queue
            .write_buffer(&self.uniform_buffers[6], 0, cast_slice(material.as_ref()));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.init.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.init
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        // draw shadow
        {
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_views[1],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            shadow_pass.set_pipeline(&self.pipelines[1]);
            shadow_pass.set_bind_group(0, &self.uniform_bind_groups[3], &[]);

            // draw shadow for cube
            shadow_pass.set_vertex_buffer(0, self.vertex_buffers[0].slice(..));
            shadow_pass
                .set_index_buffer(self.index_buffers[0].slice(..), wgpu::IndexFormat::Uint16);
            shadow_pass.draw_indexed(0..self.indices_lens[0], 0, 0..1);

            // draw shadow for torus
            shadow_pass.set_vertex_buffer(0, self.vertex_buffers[2].slice(..));
            shadow_pass
                .set_index_buffer(self.index_buffers[2].slice(..), wgpu::IndexFormat::Uint16);
            shadow_pass.draw_indexed(0..self.indices_lens[2], 0, 1..2);

            // draw shadow for spheres
            shadow_pass.set_vertex_buffer(0, self.vertex_buffers[1].slice(..));
            shadow_pass
                .set_index_buffer(self.index_buffers[1].slice(..), wgpu::IndexFormat::Uint16);
            shadow_pass.draw_indexed(0..self.indices_lens[1], 0, 2..20);
        }

        // draw objects
        {
            let color_attach = ws::create_color_attachment(&view);
            let depth_attachment =
                ws::create_depth_stencil_attachment(&self.depth_texture_views[0]);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(color_attach)],
                depth_stencil_attachment: Some(depth_attachment),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.pipelines[0]);
            render_pass.set_bind_group(0, &self.uniform_bind_groups[0], &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_groups[1], &[]);
            render_pass.set_bind_group(2, &self.uniform_bind_groups[2], &[]);

            // draw cube
            render_pass.set_vertex_buffer(0, self.vertex_buffers[0].slice(..));
            render_pass
                .set_index_buffer(self.index_buffers[0].slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.indices_lens[0], 0, 0..1);

            // draw torus
            render_pass.set_vertex_buffer(0, self.vertex_buffers[2].slice(..));
            render_pass
                .set_index_buffer(self.index_buffers[2].slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.indices_lens[2], 0, 1..2);

            // draw spheres
            render_pass.set_vertex_buffer(0, self.vertex_buffers[1].slice(..));
            render_pass
                .set_index_buffer(self.index_buffers[1].slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.indices_lens[1], 0, 2..20);
        }

        self.init.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
