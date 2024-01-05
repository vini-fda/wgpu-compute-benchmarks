extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, DeriveInput, Data, Fields};

#[proc_macro_derive(Kernel)]
pub fn kernel_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let name = &ast.ident;
    let fields = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("Kernel only works with structs with named fields"),
        },
        _ => panic!("Kernel only works with structs"),
    };

    let field_calls = fields.iter().map(|f| {
        let name = &f.ident;
        quote! {
            self.#name.add_to_pass(pass);
        }
    });

    let expanded = quote! {
        impl Kernel for #name {
            fn add_to_pass<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
                #( #field_calls )*
            }
        }
    };

    TokenStream::from(expanded)
}
