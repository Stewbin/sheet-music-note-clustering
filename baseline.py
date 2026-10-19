import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from scipy.cluster.hierarchy import linkage, dendrogram
    from clustering import featurize_handcrafted

    return


if __name__ == "__main__":
    app.run()
