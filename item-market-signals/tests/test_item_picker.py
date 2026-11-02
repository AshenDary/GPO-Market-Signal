from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard.components.views import _item_picker_context_for, _item_picker_options


def test_item_picker_options_limit_the_catalog_to_a_prefix() -> None:
    names = ["Apple", "Candy Cane", "Candy Cane Prestige", "Flower Sword"]

    assert _item_picker_options(names, "") == names
    assert _item_picker_options(names, "c") == ["Candy Cane", "Candy Cane Prestige"]
    assert _item_picker_options(names, "CAN") == ["Candy Cane", "Candy Cane Prestige"]
    assert _item_picker_options(names, "z") == []


def test_item_picker_context_requires_a_known_item() -> None:
    names = ["Apple", "Candy Cane"]

    assert _item_picker_context_for("Candy Cane", names) == "C"
    assert _item_picker_context_for("New item", names) is None
    assert _item_picker_context_for(None, names) is None


def test_item_picker_remembers_the_selected_items_catalog_area() -> None:
    script = (
        "import pandas as pd\n"
        "from dashboard.components.views import render_item_picker\n"
        "catalog = pd.DataFrame({'name': ['Apple', 'Candy Cane', 'Candy Cane Prestige', 'Flower Sword']})\n"
        "render_item_picker(catalog, 'test_item', default_to_first=False)\n"
    )

    page = AppTest.from_string(script).run(timeout=30)

    assert not page.exception
    assert page.text_input[0].value == ""
    assert page.selectbox[0].options == ["Apple", "Candy Cane", "Candy Cane Prestige", "Flower Sword"]

    page.selectbox[0].select("Candy Cane").run(timeout=30)

    assert not page.exception
    assert page.text_input[0].value == "C"
    assert page.selectbox[0].value == "Candy Cane"
    assert page.selectbox[0].options == ["Candy Cane", "Candy Cane Prestige"]
