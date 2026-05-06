import importlib


def test_calculate_index_params_import_has_no_stdout(capsys):
    import lnclite.utils.calculate_index_params as calculate_index_params

    importlib.reload(calculate_index_params)

    captured = capsys.readouterr()
    assert captured.out == ""
