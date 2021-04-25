use lyon::math::point;
use lyon::path::Path as LyonPath;
use lyon::tessellation::*;
use nalgebra_glm as glm;
use ttf_parser as ttf;

use std::collections::HashMap;
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
    hor_advance: u16,
    ver_advance: u16,
    hor_side_bearing: i16,
    ver_side_bearing: i16,
    y_origin: i16,
    bbox: ttf::Rect,
}

impl CharacterSizing {
    fn new(
        hor_advance: u16,
        ver_advance: u16,
        hor_side_bearing: i16,
        ver_side_bearing: i16,
        y_origin: i16,
        bbox: ttf::Rect,
    ) -> Self {
        return Self {
            hor_advance,
            ver_advance,
            hor_side_bearing,
            ver_side_bearing,
            y_origin,
            bbox,
        };
    }
}

pub struct Character {
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

    pub fn get_character(&self, c: char) -> Option<Character> {
        // TODO(ish): add the character into the hash map and also
        // check if character was already cached
        let face = &self.face;

        let glyph_id = match face.glyph_index(c) {
            Some(id) => id,
            None => return None,
        };

        let mut builder = Builder(LyonPath::builder());

        let bbox = match face.outline_glyph(glyph_id, &mut builder) {
            Some(bbox) => bbox,
            None => return None,
        };

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

        let character = Character::new(
            geometry,
            CharacterSizing::new(
                face.glyph_hor_advance(glyph_id).unwrap(),
                face.glyph_ver_advance(glyph_id).unwrap(),
                face.glyph_hor_side_bearing(glyph_id).unwrap(),
                face.glyph_ver_side_bearing(glyph_id).unwrap(),
                face.glyph_y_origin(glyph_id).unwrap(),
                bbox,
            ),
        );
        return Some(character);
    }
}
