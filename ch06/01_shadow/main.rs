#[path = "../../common/app.rs"]
mod app;

mod state;

use winit::event_loop::EventLoop;
use app::App;

fn main() {
    let title = "ch06 shadow mapping";
    let _ = run(title);

    pub fn run(title: &'static str) -> anyhow::Result<()> {
        env_logger::init();

        let event_loop = EventLoop::with_user_event().build()?;
        let mut app = App::new(title, 1, None);

        event_loop.run_app(&mut app)?;

        Ok(())
    }
}
