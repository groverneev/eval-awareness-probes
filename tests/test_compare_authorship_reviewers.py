from vnaa.compare_authorship_reviewers import agreement, ending_vector


def test_ending_vector_has_stable_a_b_order():
    labels = {
        "x": {"ending_a_pass": True, "ending_b_pass": False},
        "y": {"ending_a_pass": False, "ending_b_pass": True},
    }
    assert ending_vector(labels, ["x", "y"]) == [True, False, False, True]


def test_agreement_reports_confusion_and_kappa():
    result = agreement([True, True, False, False], [True, False, True, False])
    assert result["n_agree"] == 2
    assert result["agreement_rate"] == 0.5
    assert result["cohen_kappa"] == 0.0
    assert result["confusion"]["first_pass_second_fail"] == 1
    assert result["confusion"]["first_fail_second_pass"] == 1
