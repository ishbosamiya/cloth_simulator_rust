pub trait Drawable<ExtraData, Error> {
    fn draw(&self, extra_data: &mut ExtraData) -> Result<(), Error>;
    fn draw_wireframe(&self, _extra_data: &mut ExtraData) -> Result<(), Error> {
        println!("error: draw_wireframe() not implemented but called");
        return Ok(());
    }
}
