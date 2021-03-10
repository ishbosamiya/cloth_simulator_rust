pub trait Drawable<ExtraData, Error> {
    fn draw(&self, extra_data: &mut ExtraData) -> Result<(), Error>;
}
