import os

files_in_order = [
    "src/configuracoes.py",
    "src/utilidades.py",
    "src/modelos.py",
    "src/pipeline_dados/coletar_dados.py",
    "01_otimizar_hiperparametros.py",
    "02_treinamento_walk_forward.py",
    "03_avaliar_estatisticas.py",
    "04_avaliar_portfolio.py",
    "06_benchmark_custo_computacional.py",
    "05_gerar_visualizacoes.py"
]

latex_header = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{minted}
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{xcolor}

% Configuração para o minted quebrar linhas longas
\setminted{breaklines=true, breakanywhere=true, fontsize=\scriptsize, linenos=true}
\usemintedstyle{vs}

\title{\textbf{Anexo de Códigos: MSR-TCN}}
\author{Pipeline MLOps Completo}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""

latex_footer = r"""
\end{document}
"""

with open("codigos_fonte.tex", "w", encoding="utf-8") as f_out:
    f_out.write(latex_header)
    for filepath in files_in_order:
        if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            section_title = filename.replace("_", r"\_")
            f_out.write(f"\\section{{{section_title}}}\n")
            filepath_escaped = filepath.replace("_", "\\_")
            f_out.write(f"\\textbf{{Caminho do arquivo:}} \\texttt{{{filepath_escaped}}}\n\n")
            f_out.write("\\begin{minted}{python}\n")
            
            with open(filepath, "r", encoding="utf-8") as f_in:
                code_content = f_in.read()
                f_out.write(code_content)
                if not code_content.endswith("\n"):
                    f_out.write("\n")
            
            f_out.write("\\end{minted}\n\n")
            f_out.write("\\newpage\n\n")
            
    f_out.write(latex_footer)

print("Arquivo codigos_fonte.tex gerado com sucesso!")
