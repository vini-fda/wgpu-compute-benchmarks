pub struct ShaderTemplate {
    pub shader_source: String,
}

impl ShaderTemplate {
    pub fn new(shader_source: &str) -> Self {
        Self {
            shader_source: shader_source.to_string(),
        }
    }

    pub fn replace(&self, identifier: &str, replacement: &str) -> Self {
        // replace all instances of the pattern "{{ identifier }}" with replacement
        let pattern = format!("{{{{ {} }}}}", identifier);
        let new_shader_source = self.shader_source.replace(&pattern, replacement);
        Self {
            shader_source: new_shader_source,
        }
    }

    pub fn finish(self) -> String {
        self.shader_source
    }
}