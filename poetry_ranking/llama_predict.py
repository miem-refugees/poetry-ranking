import asyncio
import re

import pandas as pd
from ollama import AsyncClient
from tqdm.asyncio import tqdm as tqdm_async


def get_average_point(text):
    lines = text.splitlines()

    digits_line = ""
    for line in lines:
        if "Финальная оценка" in line:
            digits_line = line
            break
        if "Финальная оценка".lower() in line.lower():
            digits_line = line
            break
    result = (
        int(re.search(r"\d{1,2}", digits_line.replace("100", "")).group())
        if re.search(r"\d{1,2}", digits_line.replace("100", ""))
        else None
    )
    if result is None:
        raise Exception
    return result


async def chat(poems):
    semaphore = asyncio.Semaphore(5)

    async def get_response(poem):
        async with semaphore:
            prompt = f"""Проанализируй данное стихотворение и оцени его по следующим критериям, выставляя итоговую оценку от 0 до 100, где 0 — низкое качество, а 100 — высочайшее качество: Качество рифмы: оцени, насколько рифмы точны, гармоничны и уместны (0–100). Смысл и глубина: оцени, насколько содержателен текст, передает ли он глубокие идеи или оригинальные мысли, вызывает ли эмоции (0–100). Лексическое богатство и выразительность: оцени выбор слов, их сочетаемость и выразительность (0–100). Ритм и структура: оцени плавность ритма и соблюдение формы и структуры стихотворения (0–100). Общее впечатление: оцени общий эффект от стихотворения, насколько оно звучит цельно и выразительно (0–100). В конце напиши "Финальная оценка" и выведи финальную оценку от 0 до 100.

```
{poem}
```
"""
            message = {"role": "user", "content": prompt}
            for i in range(5):
                try:
                    response = await AsyncClient().chat(
                        model="llama3.2", messages=[message]
                    )
                    point = get_average_point(response["message"]["content"])
                    print(point)
                    return point
                except Exception:
                    print("Exception")
                    continue
            return None

    coroutines = [get_response(poem) for poem in poems]
    responses = await tqdm_async.gather(*coroutines)
    return responses


async def main():
    df_test = pd.read_csv("data/raw/poetry_data_test.zip")

    pushkin = """«Мой дядя самых честных правил,
Когда не в шутку занемог,
Он уважать себя заставил
И лучше выдумать не мог.
Его пример другим наука;
Но, боже мой, какая скука
С больным сидеть и день и ночь,
Не отходя ни шагу прочь!
Какое низкое коварство
Полуживого забавлять,
Ему подушки поправлять,
Печально подносить лекарство,
Вздыхать и думать про себя:
Когда же черт возьмет тебя!»"""

    esenin = """Заметался пожар голубой,
Позабылись родимые дали.
В первый раз я запел про любовь,
В первый раз отрекаюсь скандалить.
Был я весь — как запущенный сад,
Был на женщин и зелие падкий.
Разонравилось пить и плясать
И терять свою жизнь без оглядки.
Мне бы только смотреть на тебя,
Видеть глаз злато-карий омут,
И чтоб, прошлое не любя,
Ты уйти не смогла к другому."""

    gpt_random = """Скачет воробей на лужу,
Сыр с орехом в тёмном стуже.
Три колеса по лужайке плывут,
А в горле йогурт, а рядом салют.

Липнет клюква к потолку,
Жук танцует в молоку,
В небе радуга-тетрадь,
Как же тут не полетать?"""

    top_poem_in_test = df_test[df_test["rating"] == df_test["rating"].max()][
        "output_text"
    ].values[0]

    bottom_poem_in_test = df_test[df_test["views"] == df_test["views"].min()][
        "output_text"
    ].values[1]

    random_words = """Озеро
Путешествие
Сияние
Листопад
Велосипед
Мечта
Зонтик
Река
Горизонт
Вдохновение
Фонарь
Ласточка
Ступенька"""

    sanity_check = [
        pushkin,
        esenin,
        gpt_random,
        top_poem_in_test,
        bottom_poem_in_test,
        random_words,
    ]

    poems = sanity_check

    # poems = df_test["output_text"][:3000].to_list()

    answers = await chat(poems)
    with open("llama_sanity_check.txt", "w") as file:
        for item in answers:
            if item is None:
                file.write("None" + "\n")
            else:
                file.write(str(item) + "\n")


asyncio.run(main())
