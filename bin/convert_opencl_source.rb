#!/usr/bin/env ruby

# Helper functions for generating c++ code
class CPlusPlusGenerator
  attr_accessor :cpp_namespaces

  def initialize(cpp_namespaces)
    @cpp_namespaces = cpp_namespaces
  end

  def include_user_header(name)
    "#include \"#{name}\"\n"
  end

  def include_system_hedear(name)
    "#include <#{name}>\n"
  end

  def namespace_scope
    cpp_namespaces.join("::")
  end

  def namespaces_begin
    begin_nested = ""
    cpp_namespaces.each do |cpp_namespace|
      begin_nested += "namespace #{cpp_namespace} {\n"
    end
    begin_nested
  end

  def namespaces_end
    end_nested = ""
    cpp_namespaces.each do |cpp_namespace|
      end_nested += "}\n"
    end
    end_nested
  end

  def string_literal(text)
    escaped = '"'
    text.each_line do |line|
      escaped += line.chomp.gsub('"','\"' ) + "\\n"
    end
    escaped += '"'
  end
end

# OpenCL source file
class OpenCLFile
  attr_reader   :path

  def initialize(path)
    @path = path
  end

  def contents
    @contents ||= File.read(path)
  end
end

# Transform OpenCL source file into a C++ file. The C++ file has a function
# that contains the OpenCL source as a character literal.
class OpenCLToCPlusPlusTransformer
  attr_reader :opencl_file

  def initialize(opencl_file, cpp_namespaces)
    @opencl_file = opencl_file
    @generator   = CPlusPlusGenerator.new(cpp_namespaces)
  end

  def function_name
    file_name = File.basename(opencl_file.path)
    file_name.gsub('.', '_') + "__source"
  end

  def cpp_header_name
    declaration_path = "#{function_name}.h"
  end

  def cpp_source_name
    declaration_path = "#{function_name}.cpp"
  end

  def cpp_header_contents
    contents  = generator.include_system_hedear("stddef.h")
    contents += generator.namespaces_begin
    contents += cpp_function_signature + ";\n"
    contents += generator.namespaces_end
  end

  def cpp_source_contents
    contents  = generator.include_system_hedear("cstring")
    contents += generator.include_user_header(cpp_header_name)
    contents += generator.namespaces_begin
    contents += cpp_function_signature + " {\n"
    contents += "  *name = \"#{File.basename(opencl_file.path)}\";\n"
    contents += "  *data  = #{generator.string_literal(opencl_file.contents)};"
    contents += "  length = std::strlen(*data) + 1U;\n"
    contents += "  return;\n"
    contents += "}\n"
    contents += generator.namespaces_end
  end

  private
  attr_reader :generator

  def cpp_function_signature
    "void #{function_name}(const char ** name, const char ** data, size_t & length)"
  end
end

# Generate header and source files that return a path->source mapping for all OpenCL files
class CppToOpenCLSourceMapper
  attr_reader :opencl_files
  attr_reader :source_class
  attr_reader :source_class_header

  def initialize(opencl_files, cpp_namespaces, source_class, source_class_header)
    @opencl_files        = opencl_files
    @generator           = CPlusPlusGenerator.new(cpp_namespaces)
    @source_class        = source_class
    @source_class_header = source_class_header
  end

  def header_contents
    contents  = generator.include_system_hedear('map')
    contents += generator.include_user_header(source_class_header)
    contents += generator.namespaces_begin
    contents += <<FUNCTION_DECLARATION
std::map<std::string, #{source_class}> paths_to_sources();
FUNCTION_DECLARATION
    contents += generator.namespaces_end
  end

  def source_contents
    contents  = generator.include_user_header('generated_opencl_sources.h')
    contents += generator.include_system_hedear('string')
    opencl_files.each do |file|
      contents += generator.include_user_header(file.cpp_header_name)
    end

    contents += generator.namespaces_begin
    contents += <<BEGIN_FUNCTION_DEFINITION
std::map<std::string, #{source_class}> paths_to_sources() {
    std::map<std::string, #{source_class}> mapping;
    const char * original_path;
    size_t data_length(0U);
    const char * data;

BEGIN_FUNCTION_DEFINITION

    opencl_files.each do |file|
      contents += "    #{generator.namespace_scope}::#{file.function_name}(&original_path, &data, data_length);\n"
      contents += "    mapping.insert( std::pair<std::string, #{source_class}>( original_path, #{source_class}(data, data_length) ) );\n\n"
    end

    contents += <<END_FUNCTION_DEFINITION
    return mapping;
}
END_FUNCTION_DEFINITION

    contents += generator.namespaces_end
  end

  private
  attr_reader :generator
end


require 'fileutils'
require 'pathname'

if ARGV.length != 5
  STDERR.puts "usage: convert_opencl source_root_dir target_dir namespace source_class source_class_header"
  exit(1)
end
source_root_dir     = Pathname.new(ARGV[0])
target_dir          = Pathname.new(ARGV[1])
cpp_namespaces      = ARGV[2].split('::')
source_class        = ARGV[3]
source_class_header = ARGV[4]

# for all *.cl files, generate h, cpp files
opencl_file_paths = Dir.glob(source_root_dir.join("**/*.cl"))
opencl_transformers = []
opencl_file_paths.each do |opencl_path|
  opencl_file = OpenCLFile.new(opencl_path)
  transformer = OpenCLToCPlusPlusTransformer.new(opencl_file, cpp_namespaces)
  opencl_transformers << transformer

  puts "#{opencl_path} => #{transformer.cpp_source_name}, #{transformer.cpp_header_name}"
  File.open(target_dir.join(transformer.cpp_header_name), "w+").puts(transformer.cpp_header_contents)
  File.open(target_dir.join(transformer.cpp_source_name), "w+").puts(transformer.cpp_source_contents)
end

mapper = CppToOpenCLSourceMapper.new(opencl_transformers, cpp_namespaces, source_class, source_class_header)
File.open(target_dir.join("generated_opencl_sources.h"), "w+").puts(mapper.header_contents)
File.open(target_dir.join("generated_opencl_sources.cpp"), "w+").puts(mapper.source_contents)
