use app::Application;
use winit::event_loop::EventLoop;

mod app;
mod state;

fn main() {
    let title = "ch06 shadow mapping";
    let _ = run(title);

    pub fn run(title: &'static str) -> anyhow::Result<()> {
        env_logger::init();

        let event_loop = EventLoop::builder().build()?;
        let mut app = Application::new(title, 1, None);

        event_loop.run_app(&mut app)?;

        Ok(())
    }
}
