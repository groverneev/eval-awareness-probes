#!/usr/bin/env bash

set -euo pipefail

no_open=false
if [[ "${1:-}" == "--no-open" ]]; then
    no_open=true
fi

paper_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
document_name="neurips_2026"
source_file="${document_name}.tex"
build_directory="${paper_directory}/.build"
build_pdf="${build_directory}/${document_name}.pdf"
output_pdf="${paper_directory}/${document_name}.pdf"

# MacTeX installs its command-line tools here. Keeping the existing PATH as a
# fallback also supports TeX Live on Linux and custom installations.
if [[ -d /Library/TeX/texbin ]]; then
    export PATH="/Library/TeX/texbin:${PATH}"
fi

for tool in pdflatex bibtex; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "Could not find ${tool}. Install MacTeX (macOS) or TeX Live (Linux)." >&2
        exit 1
    fi
done

run_tex_command() {
    local description="$1"
    shift
    printf '\n==> %s\n' "${description}"
    "$@"
}

cd "${paper_directory}"
mkdir -p "${build_directory}"
cp "${paper_directory}/references.bib" "${build_directory}/references.bib"

pdflatex_arguments=(
    -synctex=1
    -interaction=nonstopmode
    -file-line-error
    -halt-on-error
    -output-directory="${build_directory}"
    "${source_file}"
)

run_tex_command 'Initial LaTeX pass' pdflatex "${pdflatex_arguments[@]}"
(
    cd "${build_directory}"
    run_tex_command 'Bibliography pass' bibtex "${document_name}"
)
run_tex_command 'Second LaTeX pass' pdflatex "${pdflatex_arguments[@]}"
run_tex_command 'Third LaTeX pass' pdflatex "${pdflatex_arguments[@]}"
run_tex_command 'Final stabilization pass' pdflatex "${pdflatex_arguments[@]}"

if [[ ! -f "${build_pdf}" ]]; then
    echo "The build completed without creating ${build_pdf}." >&2
    exit 1
fi

cp "${build_pdf}" "${output_pdf}"

printf '\nPaper built successfully: %s\n' "${output_pdf}"

if [[ "${no_open}" == false ]]; then
    if command -v code >/dev/null 2>&1; then
        code --reuse-window "${output_pdf}"
    elif [[ "$(uname -s)" == 'Darwin' ]]; then
        open -a 'Visual Studio Code' "${output_pdf}"
    else
        echo 'Open the generated PDF from the VS Code Explorer.'
    fi
fi
