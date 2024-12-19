#[path = "../common/app.rs"]
mod app;

#[path = "../common/state.rs"]
mod state;

use wgpu_lighting::vertex_data as vd;

use std::error::Error;
use winit::event_loop::EventLoop;

use crate::app::Application;
use crate::state::Vertex;

fn create_vertices() -> (Vec<Vertex>, Vec<u16>, Vec<u16>) {
    let (pos, norm, ind, ind2) = vd::create_torus_data(1.8, 0.4, 60, 20);
    let mut data: Vec<Vertex> = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        data.push(Vertex {
            position: pos[i],
            normal: norm[i],
        });
    }
    (data.to_vec(), ind, ind2)
}

fn main() {
    let mut sample_count = 1 as u32;
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        sample_count = args[1].parse::<u32>().unwrap();
    }

    let title = "ch01 bling phong torus";

    let (vertex_data, index_data, index_data2) = create_vertices();
    let _ = run(&vertex_data, &index_data, &index_data2, sample_count, title);

    pub fn run(
        vertex_data: &Vec<Vertex>,
        index_data: &Vec<u16>,
        index_data2: &Vec<u16>,
        sample_count: u32,
        title: &str,
    ) -> Result<(), Box<dyn Error>> {
        env_logger::init();

        let event_loop = EventLoop::builder().build()?;
        let mut app = Application::new(
            &vertex_data,
            &index_data,
            &index_data2,
            sample_count,
            title,
            None,
        );

        event_loop.run_app(&mut app)?;

        Ok(())
    }
}
