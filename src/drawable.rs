pub trait Drawable<Error> {
    fn draw(&self) -> Result<(), Error>;
}
