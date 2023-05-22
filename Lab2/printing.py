from sympy.printing.latex import LatexPrinter

class MatrixAlignRightPrinter(LatexPrinter):
    def _print_matrix_contents(self, expr):
        lines = []

        for line in range(expr.rows):  # horrible, should be "rows"
            lines.append(" & ".join([self._print(i) for i in expr[line, :]]))

        mat_str = self._settings["mat_str"]
        if mat_str is None:
            if self._settings["mode"] == "inline":
                mat_str = "smallmatrix"
            else:
                if (expr.cols <= 10) is True:
                    mat_str = "matrix"
                else:
                    mat_str = "array"

        out_str = r"\begin{%MATSTR%}%s\end{%MATSTR%}"
        out_str = out_str.replace("%MATSTR%", mat_str)

        align = "r" if expr.cols > 1 else "c"

        if mat_str == "array":
            out_str = out_str.replace("%s", "{" + align * expr.cols + "}%s")
        return out_str % r"\\".join(lines)