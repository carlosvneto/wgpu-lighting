use std::sync::Arc;
use std::time;
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::PhysicalKey,
    window::{Window, WindowId},
};

use crate::state::{State, Vertex};

pub struct App<'a> {
    state: Option<State>,
    vertex_data: &'a Vec<Vertex>,
    index_data: &'a Vec<u16>,
    index_data2: &'a Vec<u16>,
    sample_count: u32,
    title: &'static str,
    render_start_time: Option<time::Instant>,
}

impl<'a> App<'a> {
    pub fn new(
        vertex_data: &'a Vec<Vertex>,
        index_data: &'a Vec<u16>,
        index_data2: &'a Vec<u16>,
        sample_count: u32,
        title: &'static str,
        render_start_time: Option<time::Instant>,
    ) -> Self {
        Self {
            state: None,
            vertex_data,
            index_data,
            index_data2,
            sample_count,
            title,
            render_start_time,
        }
    }
}

impl<'a> ApplicationHandler<State> for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_title(self.title);

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        self.state = Some(pollster::block_on(async {
            State::new(
                window.into(),
                self.vertex_data,
                self.index_data,
                self.index_data2,
                self.sample_count,
            )
            .await
        }));
        
        self.render_start_time = Some(time::Instant::now());
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        self.state = Some(event);
    }
    
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                state.window().request_redraw();
                let now = std::time::Instant::now();
                let dt = now - self.render_start_time.unwrap_or(now);
                state.update(dt);
                match state.render() {
                    Ok(_) => {}
                    // Rebuild your Surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window().inner_size();
                        state.resize(size.width, size.height);
                    }
                    // Terminate application if memory is low
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        println!("Out of memory");
                        event_loop.exit();
                    }
                    // If a frame takes too long to display, warn and move on to the next frame
                    Err(wgpu::SurfaceError::Timeout) => {
                        println!("Surface timeout");
                    }
                    Err(wgpu::SurfaceError::Other) => {
                        println!("Surface error");
                    }  
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window().request_redraw();
        }
    }
}
