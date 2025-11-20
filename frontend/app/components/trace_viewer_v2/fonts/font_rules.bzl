"""Starlark rules for generating C++ headers from font files for ImGui."""

load(
    "//third_party/bazel_rules/rules_cc/cc/common:cc_common.bzl",
    "cc_common",
)
load(
    "//third_party/bazel_rules/rules_cc/cc/common:cc_info.bzl",
    "CcInfo",
)

def _imgui_font_headers_impl(ctx):
    output_headers = []
    tool = ctx.executable._tool
    args_str = " ".join(ctx.attr.args)
    if args_str:
        args_str += " "

    for font_file in ctx.files.srcs:
        base_name = font_file.basename.lower().removesuffix(".ttf")
        symbol_name = base_name.replace("-", "_").replace("[", "_").replace("]", "").replace(",", "")
        header = ctx.actions.declare_file(symbol_name + ".h")
        output_headers.append(header)

        ctx.actions.run_shell(
            outputs = [header],
            inputs = [font_file],
            tools = [tool],
            mnemonic = "FontHeaderBuilder",
            progress_message = "Generating header for %s" % font_file.basename,
            command = "'{tool_path}' {args}'{font_path}' {symbol} > '{header_path}'".format(
                tool_path = tool.path,
                args = args_str,
                font_path = font_file.path,
                symbol = symbol_name,
                header_path = header.path,
            ),
        )

    generated_headers_depset = depset(output_headers)

    return [
        DefaultInfo(files = generated_headers_depset),
        CcInfo(
            compilation_context = cc_common.create_compilation_context(
                headers = generated_headers_depset,
            ),
        ),
    ]

imgui_font_headers = rule(
    implementation = _imgui_font_headers_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
            doc = "A filegroup target containing the font files.",
        ),
        "args": attr.string_list(
            doc = "A list of command-line arguments to pass to the tool (e.g., ['-base85']).",
        ),
        "_tool": attr.label(
            default = Label("//third_party/dear_imgui:binary_to_compressed_c"),
            executable = True,
            cfg = "exec",
        ),
    },
)
