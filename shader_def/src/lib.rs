use glsl::syntax::{
    ArraySpecifierDimension, Expr, ExternalDeclaration, InitDeclaratorList, StorageQualifier,
    TranslationUnit, TypeQualifierSpec, TypeSpecifierNonArray,
};

use quote::format_ident;

#[proc_macro]
pub fn shaderdef_from_file(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let shader_def = syn::parse::<ShaderDef>(tokens).expect("Invalid arguments");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_path = format!("{manifest_dir}/{}", shader_def.vertex_shader.value());
    let shader = parse_shader(file_path);

    let mut uniform_definitions = VariableDefinitions::new();
    let mut vs_inputs = Vec::new();

    // process vertex shader
    for external_declaration in shader {
        let Some(decl_list) = as_init_declarator_list(external_declaration) else {
            continue;
        };
        match DeclarationType::identify(&decl_list.head) {
            DeclarationType::In => {
                let InitDeclaratorList { head, tail } = decl_list;
                let mut count = 0u8;
                if !head.name.is_none() {
                    count += 1;
                }
                count += tail.len() as u8;
                let ty = head.ty.ty.ty;
                vs_inputs.push(get_input_variant(&ty))
            }
            DeclarationType::Uniform => uniform_definitions.extend(decl_list),
            _ => (),
        }
    }

    // process fragment shader
    let file_path = format!("{manifest_dir}/{}", shader_def.fragment_shader.value());
    let shader = parse_shader(file_path);
    for external_declaration in shader {
        let Some(decl_list) = as_init_declarator_list(external_declaration) else {
            continue;
        };
        if matches!(
            DeclarationType::identify(&decl_list.head),
            DeclarationType::Uniform
        ) {
            uniform_definitions.extend(decl_list)
        }
    }

    generate_tokens(&shader_def, uniform_definitions, vs_inputs).into()
}

fn parse_shader<P: AsRef<std::path::Path> + std::fmt::Display>(file_path: P) -> TranslationUnit {
    use glsl::parser::Parse;
    let file = std::fs::File::open(&file_path).expect(&format!("Couldn't open file {file_path}"));
    let src = std::io::read_to_string(file).expect(&format!("IO error reading file {file_path}"));

    glsl::syntax::TranslationUnit::parse(src).expect(&format!("Couldn't parse shader source"))
}

fn as_init_declarator_list(declaration: ExternalDeclaration) -> Option<InitDeclaratorList> {
    use glsl::syntax::Declaration;
    match declaration {
        ExternalDeclaration::Declaration(Declaration::InitDeclaratorList(decl_list)) => {
            Some(decl_list)
        }
        _ => None,
    }
}

enum DeclarationType {
    In,
    InOut,
    Out,
    Uniform,
    Irrelevant,
}
impl DeclarationType {
    fn identify(decl: &glsl::syntax::SingleDeclaration) -> DeclarationType {
        let Some(qualifiers) = decl.ty.qualifier.as_ref().map(|q| q.qualifiers.0.iter()) else {
            return DeclarationType::Irrelevant;
        };
        for qualifier in qualifiers {
            let kind = match qualifier {
                TypeQualifierSpec::Storage(StorageQualifier::In) => DeclarationType::In,
                TypeQualifierSpec::Storage(StorageQualifier::Uniform) => DeclarationType::Uniform,
                TypeQualifierSpec::Storage(StorageQualifier::InOut) => DeclarationType::InOut,
                TypeQualifierSpec::Storage(StorageQualifier::Out) => DeclarationType::Out,
                _ => continue,
            };
            return kind;
        }
        DeclarationType::Irrelevant
    }
}

struct VariableDefinitions(Vec<DefinitionGroup>);
impl VariableDefinitions {
    fn new() -> Self {
        Self(vec![])
    }
    fn extend(&mut self, decl_list: InitDeclaratorList) {
        let InitDeclaratorList { head, tail } = decl_list;
        let type_name = TypeName::from(head.ty.ty.ty);
        if let Some(name) = head.name {
            let ty = match head.array_specifier {
                Some(spec) => match spec.dimensions.0[0] {
                    ArraySpecifierDimension::Unsized => todo!("Unsized array"),
                    ArraySpecifierDimension::ExplicitlySized(ref e) => match e.as_ref() {
                        &Expr::IntConst(n) => DataType::new(type_name, n as usize),
                        _ => todo!("Other Expr types not implemented"),
                    },
                },
                None => DataType::new(type_name, 0),
            };
            if let Some(e) = self.0.iter_mut().find(|e| e.ty == ty) {
                e.vars.push(name.0)
            } else {
                self.0.push(DefinitionGroup {
                    ty,
                    vars: vec![name.0],
                })
            }
        }
        for element in tail {
            let name = element.ident.ident.0;
            let ty = match element.ident.array_spec {
                Some(spec) => match spec.dimensions.0[0] {
                    ArraySpecifierDimension::Unsized => todo!("Unsized array"),
                    ArraySpecifierDimension::ExplicitlySized(ref e) => match e.as_ref() {
                        &Expr::IntConst(n) => DataType::new(type_name, n as usize),
                        _ => todo!("Other Expr types not implemented"),
                    },
                },
                None => DataType::new(type_name, 0),
            };
            if let Some(e) = self.0.iter_mut().find(|e| e.ty == ty) {
                e.vars.push(name)
            } else {
                self.0.push(DefinitionGroup {
                    ty,
                    vars: vec![name],
                })
            }
        }
    }
}
struct DefinitionGroup {
    ty: DataType,
    vars: Vec<String>,
}

struct ShaderDef {
    crate_path: syn::Path,
    shader_name: syn::Ident,
    vertex_shader: syn::LitStr,
    fragment_shader: syn::LitStr,
    // geometry_shader: Option<syn::LitStr>,
    // tessellation_shader: Option<(syn::LitStr, syn::LitStr)>,
}

impl syn::parse::Parse for ShaderDef {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let crate_name = input.parse::<syn::Path>()?;
        let shader_name = input.parse::<syn::Ident>()?;
        let content;
        syn::braced!(content in input);
        let mut fields = content.parse_terminated(Field::parse, syn::Token![,])?;

        let mut vertex_shader = None;
        let mut fragment_shader = None;
        while let Some(field) = fields.pop() {
            let field = field.into_value();
            match &*field.name.to_string() {
                "vertex" => vertex_shader = Some(field.file),
                "fragment" => fragment_shader = Some(field.file),
                _ => return Err(input.error("Unexpected shader type")),
            }
        }
        Ok(Self {
            crate_path: crate_name,
            shader_name,
            vertex_shader: vertex_shader.ok_or(input.error("No vertex shader specified"))?,
            fragment_shader: fragment_shader.ok_or(input.error("No vertex shader specified"))?,
        })
    }
}
struct Field {
    name: syn::Ident,
    file: syn::LitStr,
}
impl syn::parse::Parse for Field {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<syn::Token![:]>().unwrap();
        let file = input.parse()?;

        Ok(Self { name, file })
    }
}

#[derive(Debug, Clone, Copy)]
struct TypeName(&'static str);
impl TypeName {
    const VEC2: Self = Self("Vec2");
    const VEC3: Self = Self("Vec3");
    const VEC4: Self = Self("Vec4");
    const IVEC2: Self = Self("IVec2");
    const IVEC3: Self = Self("IVec3");
    const IVEC4: Self = Self("IVec4");
    const MAT4: Self = Self("Mat4");
    const SAMPLER_2D: Self = Self("Sampler2D");
    const SAMPLER_2D_ARRAY: Self = Self("Sampler2DArray");
    const SAMPLER_CUBE: Self = Self("SamplerCube");
    fn is_sampler(self) -> bool {
        self == Self::SAMPLER_2D || self == Self::SAMPLER_2D_ARRAY || self == Self::SAMPLER_CUBE
    }
}

impl PartialEq for TypeName {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl From<TypeSpecifierNonArray> for TypeName {
    fn from(value: TypeSpecifierNonArray) -> Self {
        match value {
            TypeSpecifierNonArray::Void => todo!(),
            TypeSpecifierNonArray::Bool => todo!(),
            TypeSpecifierNonArray::Int => todo!(),
            TypeSpecifierNonArray::UInt => todo!(),
            TypeSpecifierNonArray::Float => todo!(),
            TypeSpecifierNonArray::Double => todo!(),
            TypeSpecifierNonArray::Vec2 => Self::VEC2,
            TypeSpecifierNonArray::Vec3 => Self::VEC3,
            TypeSpecifierNonArray::Vec4 => Self::VEC4,
            TypeSpecifierNonArray::DVec2 => todo!(),
            TypeSpecifierNonArray::DVec3 => todo!(),
            TypeSpecifierNonArray::DVec4 => todo!(),
            TypeSpecifierNonArray::BVec2 => todo!(),
            TypeSpecifierNonArray::BVec3 => todo!(),
            TypeSpecifierNonArray::BVec4 => todo!(),
            TypeSpecifierNonArray::IVec2 => Self::IVEC2,
            TypeSpecifierNonArray::IVec3 => Self::IVEC3,
            TypeSpecifierNonArray::IVec4 => Self::IVEC4,
            TypeSpecifierNonArray::UVec2 => todo!(),
            TypeSpecifierNonArray::UVec3 => todo!(),
            TypeSpecifierNonArray::UVec4 => todo!(),
            TypeSpecifierNonArray::Mat2 => todo!(),
            TypeSpecifierNonArray::Mat3 => todo!(),
            TypeSpecifierNonArray::Mat4 => Self::MAT4,
            TypeSpecifierNonArray::Mat23 => todo!(),
            TypeSpecifierNonArray::Mat24 => todo!(),
            TypeSpecifierNonArray::Mat32 => todo!(),
            TypeSpecifierNonArray::Mat34 => todo!(),
            TypeSpecifierNonArray::Mat42 => todo!(),
            TypeSpecifierNonArray::Mat43 => todo!(),
            TypeSpecifierNonArray::DMat2 => todo!(),
            TypeSpecifierNonArray::DMat3 => todo!(),
            TypeSpecifierNonArray::DMat4 => todo!(),
            TypeSpecifierNonArray::DMat23 => todo!(),
            TypeSpecifierNonArray::DMat24 => todo!(),
            TypeSpecifierNonArray::DMat32 => todo!(),
            TypeSpecifierNonArray::DMat34 => todo!(),
            TypeSpecifierNonArray::DMat42 => todo!(),
            TypeSpecifierNonArray::DMat43 => todo!(),
            TypeSpecifierNonArray::Sampler1D => todo!(),
            TypeSpecifierNonArray::Image1D => todo!(),
            TypeSpecifierNonArray::Sampler2D => Self::SAMPLER_2D,
            TypeSpecifierNonArray::Image2D => todo!(),
            TypeSpecifierNonArray::Sampler3D => todo!(),
            TypeSpecifierNonArray::Image3D => todo!(),
            TypeSpecifierNonArray::SamplerCube => Self::SAMPLER_CUBE,
            TypeSpecifierNonArray::ImageCube => todo!(),
            TypeSpecifierNonArray::Sampler2DRect => todo!(),
            TypeSpecifierNonArray::Image2DRect => todo!(),
            TypeSpecifierNonArray::Sampler1DArray => todo!(),
            TypeSpecifierNonArray::Image1DArray => todo!(),
            TypeSpecifierNonArray::Sampler2DArray => Self::SAMPLER_2D_ARRAY,
            TypeSpecifierNonArray::Image2DArray => todo!(),
            TypeSpecifierNonArray::SamplerBuffer => todo!(),
            TypeSpecifierNonArray::ImageBuffer => todo!(),
            TypeSpecifierNonArray::Sampler2DMS => todo!(),
            TypeSpecifierNonArray::Image2DMS => todo!(),
            TypeSpecifierNonArray::Sampler2DMSArray => todo!(),
            TypeSpecifierNonArray::Image2DMSArray => todo!(),
            TypeSpecifierNonArray::SamplerCubeArray => todo!(),
            TypeSpecifierNonArray::ImageCubeArray => todo!(),
            TypeSpecifierNonArray::Sampler1DShadow => todo!(),
            TypeSpecifierNonArray::Sampler2DShadow => todo!(),
            TypeSpecifierNonArray::Sampler2DRectShadow => todo!(),
            TypeSpecifierNonArray::Sampler1DArrayShadow => todo!(),
            TypeSpecifierNonArray::Sampler2DArrayShadow => todo!(),
            TypeSpecifierNonArray::SamplerCubeShadow => todo!(),
            TypeSpecifierNonArray::SamplerCubeArrayShadow => todo!(),
            TypeSpecifierNonArray::ISampler1D => todo!(),
            TypeSpecifierNonArray::IImage1D => todo!(),
            TypeSpecifierNonArray::ISampler2D => todo!(),
            TypeSpecifierNonArray::IImage2D => todo!(),
            TypeSpecifierNonArray::ISampler3D => todo!(),
            TypeSpecifierNonArray::IImage3D => todo!(),
            TypeSpecifierNonArray::ISamplerCube => todo!(),
            TypeSpecifierNonArray::IImageCube => todo!(),
            TypeSpecifierNonArray::ISampler2DRect => todo!(),
            TypeSpecifierNonArray::IImage2DRect => todo!(),
            TypeSpecifierNonArray::ISampler1DArray => todo!(),
            TypeSpecifierNonArray::IImage1DArray => todo!(),
            TypeSpecifierNonArray::ISampler2DArray => todo!(),
            TypeSpecifierNonArray::IImage2DArray => todo!(),
            TypeSpecifierNonArray::ISamplerBuffer => todo!(),
            TypeSpecifierNonArray::IImageBuffer => todo!(),
            TypeSpecifierNonArray::ISampler2DMS => todo!(),
            TypeSpecifierNonArray::IImage2DMS => todo!(),
            TypeSpecifierNonArray::ISampler2DMSArray => todo!(),
            TypeSpecifierNonArray::IImage2DMSArray => todo!(),
            TypeSpecifierNonArray::ISamplerCubeArray => todo!(),
            TypeSpecifierNonArray::IImageCubeArray => todo!(),
            TypeSpecifierNonArray::AtomicUInt => todo!(),
            TypeSpecifierNonArray::USampler1D => todo!(),
            TypeSpecifierNonArray::UImage1D => todo!(),
            TypeSpecifierNonArray::USampler2D => todo!(),
            TypeSpecifierNonArray::UImage2D => todo!(),
            TypeSpecifierNonArray::USampler3D => todo!(),
            TypeSpecifierNonArray::UImage3D => todo!(),
            TypeSpecifierNonArray::USamplerCube => todo!(),
            TypeSpecifierNonArray::UImageCube => todo!(),
            TypeSpecifierNonArray::USampler2DRect => todo!(),
            TypeSpecifierNonArray::UImage2DRect => todo!(),
            TypeSpecifierNonArray::USampler1DArray => todo!(),
            TypeSpecifierNonArray::UImage1DArray => todo!(),
            TypeSpecifierNonArray::USampler2DArray => todo!(),
            TypeSpecifierNonArray::UImage2DArray => todo!(),
            TypeSpecifierNonArray::USamplerBuffer => todo!(),
            TypeSpecifierNonArray::UImageBuffer => todo!(),
            TypeSpecifierNonArray::USampler2DMS => todo!(),
            TypeSpecifierNonArray::UImage2DMS => todo!(),
            TypeSpecifierNonArray::USampler2DMSArray => todo!(),
            TypeSpecifierNonArray::UImage2DMSArray => todo!(),
            TypeSpecifierNonArray::USamplerCubeArray => todo!(),
            TypeSpecifierNonArray::UImageCubeArray => todo!(),
            TypeSpecifierNonArray::Struct(_) => todo!(),
            TypeSpecifierNonArray::TypeName(_) => todo!(),
        }
    }
}
#[derive(Debug, PartialEq)]
struct DataType {
    name: TypeName,
    elements_count: usize,
}
impl DataType {
    fn new(specifier: TypeName, elements_count: usize) -> Self {
        Self {
            name: specifier,
            elements_count,
        }
    }
    fn as_mangled_ident(&self) -> syn::Ident {
        if self.elements_count > 0 {
            let encoded_type = Self::mangle_type_name(self.name.0);
            let array_size = self.elements_count;
            format_ident!("_{encoded_type}_{array_size}")
        } else {
            let encoded_type = Self::mangle_type_name(self.name.0);
            format_ident!("_{encoded_type}")
        }
    }
    fn mangle_type_name(ty: &str) -> String {
        let mut output = String::with_capacity(ty.len());
        for ch in ty.as_bytes() {
            let symbol_index = match ch {
                b'_' => 0,
                b'0'..b'9' => ch - 48 + 1,  // offset by 1 for the '_'
                b'A'..b'Z' => ch - 65 + 11, // offset by 11 for the previous sets of symbols
                b'a'..b'z' => ch - 97 + 35, // offset by 35 for the previous sets
                _ => panic!("Identifier can only contain characters a..z, A..Z, 0..9 and _"),
            };
            output.push_str(&format!("{:02o}", symbol_index))
        }
        output
    }
    fn as_token_stream(&self, qualifier: &syn::Path) -> quote::__private::TokenStream {
        let ty_name = format_ident!("{}", self.name.0);
        if self.elements_count > 0 {
            let n = self.elements_count;
            quote::quote! {[#qualifier::#ty_name; #n]}
        } else {
            quote::quote!(#qualifier::#ty_name)
        }
    }
    fn is_sampler_type(&self) -> bool {
        self.name.is_sampler()
    }
}

struct AttrDef {
    ty: AttrTy,
    size: u8,
    count: u8,
}

enum InputType {
    Bool,
    Int,
    UInt,
    Float,
    Double,

    BVec2,
    BVec3,
    BVec4,

    IVec2,
    IVec3,
    IVec4,

    UVec2,
    UVec3,
    UVec4,

    Vec2,
    Vec3,
    Vec4,

    DVec2,
    DVec3,
    DVec4,

    Mat2,
    Mat2x3,
    Mat2x4,
    Mat3,
    Mat3x2,
    Mat3x4,
    Mat4,
    Mat4x2,
    Mat4x3,

    DMat2,
    DMat2x3,
    DMat2x4,
    DMat3,
    DMat3x2,
    DMat3x4,
    DMat4,
    DMat4x2,
    DMat4x3,
}

impl InputType {
    fn to_str(&self) -> &'static str {
        match self {
            InputType::Bool => "Bool",
            InputType::Int => "Int",
            InputType::UInt => "UInt",
            InputType::Float => "Float",
            InputType::Double => "Double",

            InputType::BVec2 => "BVec2",
            InputType::BVec3 => "BVec3",
            InputType::BVec4 => "BVec4",

            InputType::IVec2 => "IVec2",
            InputType::IVec3 => "IVec3",
            InputType::IVec4 => "IVec4",

            InputType::UVec2 => "UVec2",
            InputType::UVec3 => "UVec3",
            InputType::UVec4 => "UVec4",

            InputType::Vec2 => "Vec2",
            InputType::Vec3 => "Vec3",
            InputType::Vec4 => "Vec4",

            InputType::DVec2 => "DVec2",
            InputType::DVec3 => "DVec3",
            InputType::DVec4 => "DVec4",

            InputType::Mat2 => "Mat2",
            InputType::Mat2x3 => "Mat2x3",
            InputType::Mat2x4 => "Mat2x4",
            InputType::Mat3 => "Mat3",
            InputType::Mat3x2 => "Mat3x2",
            InputType::Mat3x4 => "Mat3x4",
            InputType::Mat4 => "Mat4",
            InputType::Mat4x2 => "Mat4x2",
            InputType::Mat4x3 => "Mat4x3",

            InputType::DMat2 => "DMat2",
            InputType::DMat2x3 => "DMat2x3",
            InputType::DMat2x4 => "DMat2x4",
            InputType::DMat3 => "DMat3",
            InputType::DMat3x2 => "DMat3x2",
            InputType::DMat3x4 => "DMat3x4",
            InputType::DMat4 => "DMat4",
            InputType::DMat4x2 => "DMat4x2",
            InputType::DMat4x3 => "DMat4x3",
        }
    }
}

enum AttrTy {
    F32,
    I32,
    U32,
}
impl AttrTy {
    fn to_str(&self) -> &'static str {
        match self {
            AttrTy::F32 => "F32",
            AttrTy::I32 => "I32",
            AttrTy::U32 => "U32",
        }
    }
}

fn get_input_variant(ty: &glsl::syntax::TypeSpecifierNonArray) -> InputType {
    use glsl::syntax::TypeSpecifierNonArray;
    match ty {
        TypeSpecifierNonArray::Vec2 => InputType::Vec2,
        TypeSpecifierNonArray::Vec3 => InputType::Vec3,
        TypeSpecifierNonArray::Vec4 => InputType::Vec4,
        TypeSpecifierNonArray::IVec4 => InputType::IVec4,
        x => todo!("{:?}", x),
    }
}

fn generate_tokens(
    shader_def: &ShaderDef,
    defs: VariableDefinitions,
    vs_inputs: Vec<InputType>,
) -> quote::__private::TokenStream {
    let crate_path = &shader_def.crate_path;
    let struct_name = &shader_def.shader_name;

    let mut uniform_store_impls = quote::quote!();
    let mut field_definitions = quote::quote! {};
    let mut field_initialization_code = quote::quote! {};
    let mut struct_initializers = quote::quote!();
    let mut const_vars = quote::quote!();

    let mut next_sampler_idx = 0u32;

    for DefinitionGroup { ty, vars } in &defs.0 {
        let field = ty.as_mangled_ident();
        let ty_name = ty.as_token_stream(crate_path);
        let size = vars.len();
        uniform_store_impls.extend(quote::quote! {
            impl #crate_path::uniform::UniformStore<#ty_name> for #struct_name {
                fn get_store(&self) -> &[#crate_path::uniform::UniformLocation] {
                    &self.#field
                }
            }
        });
        field_definitions.extend(quote::quote! {
            #field: [#crate_path::uniform::UniformLocation; #size],
        });
        field_initialization_code.extend(quote::quote! {
            let mut #field = [#crate_path::uniform::UniformLocation::new(0i32); #size];
            let uniform_var_names = &[#(#vars),*];
            for (i, name) in #field.iter_mut().enumerate() {
                let location = program.get_uniform_location(uniform_var_names[i]);
                if location.is_valid() {
                    *name = location;
                } else {
                    panic!("Uniform variable '{}' not found! Please make sure you're loading the same shader files and the variable hasn't been optimized out.", uniform_var_names[i]);
                }
            }
        });
        struct_initializers.extend(quote::quote! {#field,});
        let store_idx = 0..vars.len();
        let consts = vars.iter().map(|var| format_ident!("{var}"));
        if ty.is_sampler_type() {
            let sampler_idx = next_sampler_idx..next_sampler_idx + vars.len() as u32;
            const_vars.extend(quote::quote! {
                #(
                    #[allow(non_upper_case_globals)]
                    pub const #consts: #crate_path::uniform::SamplerUniform<#struct_name, #ty_name> = #crate_path::uniform::SamplerUniform::new(#store_idx, #sampler_idx);
                )*
            });
            next_sampler_idx += vars.len() as u32;
        } else {
            const_vars.extend(quote::quote! {
                #(
                    #[allow(non_upper_case_globals)]
                    pub const #consts: #crate_path::uniform::Uniform<#struct_name, #ty_name> = #crate_path::uniform::Uniform::new(#store_idx);
                )*
            });
        }
    }

    let input_variants = vs_inputs
        .iter()
        .map(|input_type| format_ident!("{}", input_type.to_str()));

    quote::quote! {
        pub struct #struct_name {
            program: #crate_path::shader::ShaderProgram,
            #field_definitions
        }
        impl #struct_name {
            #const_vars
            pub fn new<P>(vertex_shader_path: &P, fragment_shader_path: &P) -> Self
            where
                P: AsRef<::std::path::Path> + ?Sized
            {
                let vs = #crate_path::shader::ShaderObject::new(
                    vertex_shader_path,
                    #crate_path::shader::ShaderType::VertexShader,
                )
                .expect(vertex_shader_path.as_ref().to_str().unwrap());
                let fs = #crate_path::shader::ShaderObject::new(
                    fragment_shader_path,
                    #crate_path::shader::ShaderType::FragmentShader,
                )
                .expect(fragment_shader_path.as_ref().to_str().unwrap());
                let program = #crate_path::shader::ShaderProgram::new(&[vs, fs]).unwrap();
                #field_initialization_code
                Self {
                    program,
                    #struct_initializers
                }
            }
        }
        impl #crate_path::ShaderInputsLayout for #struct_name {
            const LAYOUT: &'static [#crate_path::shader::InputType] = &[#(#crate_path::shader::InputType::#input_variants),*];
        }
        impl #crate_path::Bind for #struct_name
        {
            type BoundObject<'a> = #crate_path::shader::ActiveShader<'a, #struct_name>;
            fn bind<'a>(&'a mut self) -> Self::BoundObject<'a> {
                self.program.activate();
                let active_shader = #crate_path::shader::ActiveShader::new(self, <Self as #crate_path::ShaderInputsLayout>::LAYOUT);
                active_shader
            }
        }
        #uniform_store_impls
    }
}

// fn inputs_list(inputs: Vec<InputType>) -> (Vec<u8>, Vec<syn::Ident>) {
//     let mut v1 = vec![];
//     let mut v2 = vec![];

//     for input in inputs {
//         for _ in 0..input.count {
//             v1.push(input.size);
//             v2.push(format_ident!("{}", input.ty.to_str()));
//         }
//     }
//     (v1, v2)
// }
