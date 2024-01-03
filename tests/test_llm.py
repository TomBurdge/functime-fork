# mocking taken from here https://github.com/openai/openai-python/issues/715
import datetime
import os
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice

from functime.forecasting import knn

os.environ["OPENAI_API_KEY"] = "sk-..."


def create_chat_completion(response: str, role: str = "assistant") -> ChatCompletion:
    return ChatCompletion(
        id="foo",
        model="gpt-3.5-turbo",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=response,
                    role=role,
                ),
            )
        ],
        created=int(datetime.datetime.now().timestamp()),
    )


@pytest.fixture
def commodities_pred(commodities):
    y_train, y_test, test_size, freq = commodities
    forecaster = knn(freq="1mo", lags=24)
    forecaster.fit(y=y_train)
    y_pred = forecaster.predict(fh=test_size)
    return y_pred


@patch("openai.resources.chat.Completions.create")
def test_llm_analyze(openai_create, commodities_pred):
    EXPECTED_RESPONSE = "Test passed, the mock is working! ;)"
    openai_create.return_value = create_chat_completion(EXPECTED_RESPONSE)

    os.environ["OPENAI_API_KEY"] = "sk-..."

    response = commodities_pred.llm.analyze(
        context="This dataset comprises of forecasted commodity prices between 2020 to 2023.",
        basket=["Aluminum", "Banana, Europe"],
    )
    assert response == EXPECTED_RESPONSE


@patch("openai.resources.chat.Completions.create")
def test_llm_compare(openai_create, commodities_pred):
    EXPECTED_RESPONSE = "Test passed, the mock is working! ;)"
    openai_create.return_value = create_chat_completion(EXPECTED_RESPONSE)

    basket_a = ["Aluminum", "Banana, Europe"]
    basket_b = ["Chicken", "Cocoa"]

    response = commodities_pred.llm.compare(basket=basket_a, other_basket=basket_b)
    assert response == EXPECTED_RESPONSE


# tests to add:
# fails on wrong model
# fails on too long
# specify the wrong model
# specify a different model (somehow get the model attribute)
#

# to add (features):
# token limit test
# model limit test
