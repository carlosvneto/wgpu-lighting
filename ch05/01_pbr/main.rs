#[path = "../../common/app.rs"]
mod app;

mod state;

use winit::event_loop::EventLoop;
use app::App;

fn main() {
    let mut sample_count = 1 as u32;
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        sample_count = args[1].parse::<u32>().unwrap();
    }

    let title = "ch05 pbr";
    let _ = run(title, sample_count);

    pub fn run(title: &'static str, sample_count: u32) -> anyhow::Result<()> {
        env_logger::init();

        let event_loop = EventLoop::with_user_event().build()?;
        
        let mut app = App::new(title, sample_count, None);

        event_loop.run_app(&mut app)?;

        Ok(())
    }
}
