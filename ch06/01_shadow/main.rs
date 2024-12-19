use app::Application;
use std::error::Error;
use winit::event_loop::EventLoop;

mod app;
mod state;

fn main() {
    let title = "ch06 shadow mapping";
    let _ = run(title);

    pub fn run(title: &str) -> Result<(), Box<dyn Error>> {
        env_logger::init();

        let event_loop = EventLoop::builder().build()?;
        let mut app = Application::new(title, 1, None);

        event_loop.run_app(&mut app)?;

        Ok(())
    }
}
