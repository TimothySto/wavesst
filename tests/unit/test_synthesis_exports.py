import wavesst


def test_make_chirp_exported():
    assert hasattr(wavesst, 'make_chirp')
    assert callable(wavesst.make_chirp)


def test_make_amfm_exported():
    assert hasattr(wavesst, 'make_amfm')
    assert callable(wavesst.make_amfm)


def test_make_noise_exported():
    assert hasattr(wavesst, 'make_noise')
    assert callable(wavesst.make_noise)


def test_synthesis_subpackage_importable():
    import wavesst.synthesis
    assert callable(wavesst.synthesis.make_chirp)
