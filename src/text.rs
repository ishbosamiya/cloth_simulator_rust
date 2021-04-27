use lyon::math::point;
use lyon::path::Path as LyonPath;
use lyon::tessellation::*;
use nalgebra_glm as glm;
use ttf_parser as ttf;

use std::collections::HashMap;
use std::convert::TryInto;
use std::path::Path as StdPath;

struct Builder(lyon::path::path::Builder);

impl ttf::OutlineBuilder for Builder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.begin(point(x, y));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to(point(x, y));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.0.quadratic_bezier_to(point(x1, y1), point(x, y));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.0
            .cubic_bezier_to(point(x1, y1), point(x2, y2), point(x, y));
    }

    fn close(&mut self) {
        self.0.close();
    }
}

struct CharacterSizing {
    hor_advance: TextSizeFUnits,
    _ver_advance: Option<TextSizeFUnits>,
    _hor_side_bearing: Option<TextSizeFUnits>,
    _ver_side_bearing: Option<TextSizeFUnits>,
    _y_origin: Option<TextSizeFUnits>,
    _bbox: TextRectFUnits,
}

impl CharacterSizing {
    fn new(
        hor_advance: TextSizeFUnits,
        _ver_advance: Option<TextSizeFUnits>,
        _hor_side_bearing: Option<TextSizeFUnits>,
        _ver_side_bearing: Option<TextSizeFUnits>,
        _y_origin: Option<TextSizeFUnits>,
        _bbox: TextRectFUnits,
    ) -> Self {
        return Self {
            hor_advance,
            _ver_advance,
            _hor_side_bearing,
            _ver_side_bearing,
            _y_origin,
            _bbox,
        };
    }

    fn get_hor_advance(&self) -> TextSizeFUnits {
        return self.hor_advance;
    }

    fn _get_ver_advance(&self) -> Option<TextSizeFUnits> {
        return self._ver_advance;
    }

    fn _get_hor_side_bearing(&self) -> Option<TextSizeFUnits> {
        return self._hor_side_bearing;
    }

    fn _get_ver_side_bearing(&self) -> Option<TextSizeFUnits> {
        return self._ver_side_bearing;
    }

    fn _get_y_origin(&self) -> Option<TextSizeFUnits> {
        return self._y_origin;
    }

    fn _get_bbox(&self) -> TextRectFUnits {
        return self._bbox;
    }
}

struct Character {
    mesh: VertexBuffers<glm::Vec3, u32>,
    sizing: CharacterSizing,
}

impl Character {
    fn new(mesh: VertexBuffers<glm::Vec3, u32>, sizing: CharacterSizing) -> Self {
        return Self { mesh, sizing };
    }
}

pub struct Font<'a> {
    face: ttf::Face<'a>,
    char_map: HashMap<char, Character>,
}

impl<'a> Font<'a> {
    pub fn new(font_file: &'a [u8]) -> Self {
        let face = ttf::Face::from_slice(&font_file, 0)
            .expect("error: Given font file couldn't be parsed");

        return Self {
            face,
            char_map: HashMap::new(),
        };
    }

    pub fn load_font_file<P>(path: P) -> Vec<u8>
    where
        P: AsRef<StdPath>,
    {
        return std::fs::read(&path).expect("error: Path to font didn't exist");
    }

    fn get_character(&mut self, c: char) -> Option<&Character> {
        // if character was cached, return it
        if self.char_map.contains_key(&c) {
            return self.char_map.get(&c);
        }
        let face = &self.face;

        // get the glyph index for the given character
        let glyph_id = match face.glyph_index(c) {
            Some(id) => id,
            None => return None,
        };

        // build the outline of the glyph
        // TODO(ish): add support for svg glyphs and figure out what
        // to do for rasterized glyphs
        let mut builder = Builder(LyonPath::builder());
        let bbox = match face.outline_glyph(glyph_id, &mut builder) {
            Some(bbox) => bbox,
            None => return None,
        };
        let bbox = TextRectFUnits {
            x_min: TextSizeFUnits(bbox.x_min.into()),
            x_max: TextSizeFUnits(bbox.x_max.into()),
            y_min: TextSizeFUnits(bbox.y_min.into()),
            y_max: TextSizeFUnits(bbox.y_max.into()),
        };

        // tessellate the outline
        let path = builder.0.build();
        let mut tessellator = FillTessellator::new();
        let mut geometry: VertexBuffers<glm::Vec3, u32> = VertexBuffers::new();
        tessellator
            .tessellate_path(
                &path,
                &FillOptions::default(),
                &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex| {
                    let pos = vertex.position();
                    return glm::vec3(pos.x, pos.y, 0.0);
                }),
            )
            .unwrap();

        // setup character information for later usage
        let character = Character::new(
            geometry,
            CharacterSizing::new(
                TextSizeFUnits(face.glyph_hor_advance(glyph_id).unwrap().into()),
                option_funits_from_option_u16(face.glyph_ver_advance(glyph_id)),
                option_funits_from_option_i16(face.glyph_hor_side_bearing(glyph_id)),
                option_funits_from_option_i16(face.glyph_ver_side_bearing(glyph_id)),
                option_funits_from_option_i16(face.glyph_y_origin(glyph_id)),
                bbox,
            ),
        );

        // add character to cache and return it
        self.char_map.insert(c, character);
        return self.char_map.get(&c);
    }
}

pub struct Text {}

impl Text {
    pub fn render(
        string: &str,
        font: &mut Font,
        size: TextSizePT,
        position: &glm::Vec2,
        dpi: TextSizePT,
    ) {
        let mut character_pos_map: HashMap<char, Vec<TextSizeFUnits>> = HashMap::new();
        let mut current_pos = TextSizeFUnits(0.0);
        for c in string.chars() {
            let font_char = font.get_character(c).unwrap();
            let poses = character_pos_map.entry(c).or_insert(Vec::new());
            poses.push(current_pos);
            current_pos.0 += font_char.sizing.get_hor_advance().0;
        }

        let units_per_em = TextSizeFUnits(font.face.units_per_em().unwrap().into());
        let px_multiplier = funits_to_px_multiplier(size, dpi, units_per_em);
        let gl_coord_multiplier = 1.0; // TODO(ish): implement this

        let mut final_pos_map: HashMap<char, Vec<glm::Vec3>> = HashMap::new();
        for (c, poses) in character_pos_map {
            let final_poses = final_pos_map.entry(c).or_insert(Vec::new());
            for p in poses {
                let final_pos = glm::vec3(
                    funits_to_px(p, px_multiplier).0 * gl_coord_multiplier + position[0],
                    position[1],
                    0.0,
                );
                final_poses.push(final_pos);
            }
        }

        for (c, poses) in &final_pos_map {
            let mesh = &font.char_map.get(&c).unwrap().mesh;

            let mut model_matrices = Vec::new();
            for p in poses {
                let model = glm::identity();
                // let model = glm::scale(
                //     &glm::identity(),
                //     &glm::vec3(px_multiplier, px_multiplier, px_multiplier),
                // );
                let model = glm::translate(&model, &p);
                println!("c: {}", c);
                println!("p: {}", p);
                println!("model: {}", model);
                model_matrices.push(model);
            }

            let mut vao: gl::types::GLuint = 0;
            {
                let mut model_matrices_buffer: gl::types::GLuint = 0;
                let mut vbo: gl::types::GLuint = 0;
                let mut ebo: gl::types::GLuint = 0;

                unsafe {
                    gl::GenVertexArrays(1, &mut vao);
                    gl::GenBuffers(1, &mut vbo);
                    gl::GenBuffers(1, &mut ebo);

                    gl::BindVertexArray(vao);

                    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
                    gl::BufferData(
                        gl::ARRAY_BUFFER,
                        (mesh.vertices.len() * std::mem::size_of::<glm::Vec3>())
                            .try_into()
                            .unwrap(),
                        mesh.vertices.as_ptr() as *const gl::types::GLvoid,
                        gl::STATIC_DRAW,
                    );

                    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
                    gl::BufferData(
                        gl::ELEMENT_ARRAY_BUFFER,
                        (mesh.indices.len() * std::mem::size_of::<u32>())
                            .try_into()
                            .unwrap(),
                        mesh.indices.as_ptr() as *const gl::types::GLvoid,
                        gl::STATIC_DRAW,
                    );

                    // position in the shader
                    gl::EnableVertexAttribArray(0);
                    gl::VertexAttribPointer(
                        0,
                        3,
                        gl::FLOAT,
                        gl::FALSE,
                        std::mem::size_of::<glm::Vec3>().try_into().unwrap(),
                        std::ptr::null(),
                    );
                }
                unsafe {
                    gl::GenBuffers(1, &mut model_matrices_buffer);
                    gl::BindBuffer(gl::ARRAY_BUFFER, model_matrices_buffer);
                    gl::BufferData(
                        gl::ARRAY_BUFFER,
                        (model_matrices.len() * std::mem::size_of::<glm::Mat4>())
                            .try_into()
                            .unwrap(),
                        model_matrices.as_ptr() as *const gl::types::GLvoid,
                        gl::STATIC_DRAW,
                    );

                    // model matrix in the shader
                    // set is as 4 vec4's
                    gl::EnableVertexAttribArray(1);
                    gl::VertexAttribPointer(
                        1,
                        4,
                        gl::FLOAT,
                        gl::FALSE,
                        std::mem::size_of::<glm::Mat4>().try_into().unwrap(),
                        std::ptr::null::<gl::types::GLvoid>()
                            .offset((0 * std::mem::size_of::<glm::Vec4>()).try_into().unwrap()),
                    );
                    gl::EnableVertexAttribArray(2);
                    gl::VertexAttribPointer(
                        2,
                        4,
                        gl::FLOAT,
                        gl::FALSE,
                        std::mem::size_of::<glm::Mat4>().try_into().unwrap(),
                        std::ptr::null::<gl::types::GLvoid>()
                            .offset((1 * std::mem::size_of::<glm::Vec4>()).try_into().unwrap()),
                    );
                    gl::EnableVertexAttribArray(3);
                    gl::VertexAttribPointer(
                        3,
                        4,
                        gl::FLOAT,
                        gl::FALSE,
                        std::mem::size_of::<glm::Mat4>().try_into().unwrap(),
                        std::ptr::null::<gl::types::GLvoid>()
                            .offset((2 * std::mem::size_of::<glm::Vec4>()).try_into().unwrap()),
                    );
                    gl::EnableVertexAttribArray(4);
                    gl::VertexAttribPointer(
                        4,
                        4,
                        gl::FLOAT,
                        gl::FALSE,
                        std::mem::size_of::<glm::Mat4>().try_into().unwrap(),
                        std::ptr::null::<gl::types::GLvoid>()
                            .offset((3 * std::mem::size_of::<glm::Vec4>()).try_into().unwrap()),
                    );

                    gl::VertexAttribDivisor(1, 1);
                    gl::VertexAttribDivisor(2, 1);
                    gl::VertexAttribDivisor(3, 1);
                    gl::VertexAttribDivisor(4, 1);
                }
            }
            {
                unsafe {
                    gl::BindVertexArray(vao);
                    gl::DrawElementsInstanced(
                        gl::TRIANGLES,
                        mesh.indices.len().try_into().unwrap(),
                        gl::UNSIGNED_INT,
                        std::ptr::null(),
                        model_matrices.len().try_into().unwrap(),
                    );
                    gl::BindVertexArray(0);
                }
            }
        }

        // for (c, poses) in final_pos_map {
        //     print!("{}: ", c);
        //     for p in poses {
        //         print!("{} ", p[0]);
        //     }
        //     println!("");
        // }
    }
}

pub trait TextSize {}

/// Text size in `pt`, size in points where 72pt = 1 inch
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TextSizePT(pub f32);
impl TextSize for TextSizePT {}

/// Text size in `px`, size in pixels
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TextSizePX(pub f32);
impl TextSize for TextSizePX {}

/// Text size in `FUnits`. `FUnit` is the smallest measurable unit in
/// the em square, an imaginary square that is used to size and align
/// glyphs. The dimensions of the em square typically are those of the
/// full body height of a font plus some extra spacing to prevent
/// lines of text from colliding when typeset without extra leading.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TextSizeFUnits(pub f32);
impl TextSize for TextSizeFUnits {}
fn option_funits_from_option_u16(val: Option<u16>) -> Option<TextSizeFUnits> {
    match val {
        Some(v) => return Some(TextSizeFUnits(v.into())),
        None => return None,
    }
}
fn option_funits_from_option_i16(val: Option<i16>) -> Option<TextSizeFUnits> {
    match val {
        Some(v) => return Some(TextSizeFUnits(v.into())),
        None => return None,
    }
}

/// Text Rectangle
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TextRect<T: TextSize> {
    pub x_min: T,
    pub y_min: T,
    pub x_max: T,
    pub y_max: T,
}

pub type TextRectFUnits = TextRect<TextSizeFUnits>;

/// Convert funits to px
/// `multiplier` got from `funits_to_px_multiplier`
/// `multipler` made separate for optimization reasons, `multipler`
/// can be precomputed
#[inline(always)]
fn funits_to_px(from: TextSizeFUnits, multiplier: f32) -> TextSizePX {
    let from = from.0;

    return TextSizePX(from * multiplier);
}

/// Get multiplier needed to convert funits to px
fn funits_to_px_multiplier(
    point_size: TextSizePT,
    dpi: TextSizePT,
    units_per_em: TextSizeFUnits,
) -> f32 {
    let point_size = point_size.0;
    let dpi = dpi.0;
    let units_per_em = units_per_em.0;
    return point_size * dpi / (72.0 * units_per_em);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_temp() {
        let font_file = Font::load_font_file("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf");
        let mut font = Font::new(&font_file);
        Text::render(
            "asdf",
            &mut font,
            TextSizePT(5.0),
            &glm::vec2(0.0, 0.0),
            TextSizePT(72.0),
        );
    }
}
