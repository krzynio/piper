"""Polish given names, so that the English detector leaves them alone.

The detector (lid/detect.py) treats a capitalised word in the middle of a sentence as a probable foreign proper noun
and may peel a Polish ending off it: "Kasia" -> English "Kas" + "-ia" read as "Kesja", likewise "Basia", "Antoni",
"Helena". eSpeak-pl reads all of these correctly by itself, so frontend.english_words() skips them. Common names and
their inflected forms (case endings, vocative) are generated from the lists below.
"""
FEMALE = """Anna Ania Anka Aneta Agnieszka Agata Aleksandra Ola Alicja Ala Barbara Basia Beata Bożena Danuta Dorota Ewa Ewelina
Elżbieta Ela Gosia Małgorzata Grażyna Halina Hanna Hania Helena Irena Iwona Izabela Iza Jadwiga Janina Joanna Asia Jolanta
Julia Justyna Kamila Karolina Kasia Katarzyna Klaudia Krystyna Kinga Laura Lidia Magda Magdalena Maja Marta Marzena Maria
Marysia Marlena Milena Monika Natalia Nina Olga Oliwia Patrycja Paulina Renata Róża Sabina Sylwia Stanisława Stasia Teresa
Urszula Ula Weronika Wiktoria Zofia Zosia Zuzanna Zuzia Dominika Edyta Emilia Eliza Gabriela Gabrysia Honorata Ilona Inga
Józefa Kornelia Lena Lucyna Ludmiła Marianna Mirosława Nadia Paula Roksana Sandra Sara Sonia Tamara Wanda Wioletta Wiesława
Ewunia Madzia Agusia Kasieńka Martusia Olenka Gienia Hela Jasia Lusia Mania Tosia Frania Pimpusia""".split()
MALE = """Adam Adrian Albert Aleksander Andrzej Antoni Artur Bartek Bartosz Bogdan Bolesław Bronisław Cezary Czesław Damian
Daniel Dariusz Dawid Dominik Edward Emil Erwin Eryk Filip Franciszek Gabriel Grzegorz Henryk Hubert Igor Ireneusz Jacek
Jakub Jan Janusz Jarek Jarosław Jerzy Józef Julian Kacper Kamil Karol Kazimierz Konrad Krystian Krzysiek Krzysztof Leszek
Lech Leon Lucjan Łukasz Maciej Marcin Marek Marian Mariusz Mateusz Michał Mieczysław Mikołaj Mirosław Norbert Olek Oskar
Patryk Paweł Piotr Przemysław Radek Rafał Robert Roman Ryszard Sebastian Sławek Stanisław Stefan Sylwester Szymon Tadeusz
Tomasz Tomek Waldemar Wacław Walenty Wiktor Witold Władysław Włodzimierz Wojciech Zbigniew Zdzisław Zenon Zygmunt Kuba
Staszek Jasiek Michałek Wojtek Rysiek Zbyszek Mietek Heniek Józek Franek Antek Bolek Felek Kazik Lolek Stasiek""".split()

_A_ENDINGS = ("a", "y", "i", "ę", "ą", "o", "e", "ie", "ii", "iom", "om", "ami", "ach")
_C_ENDINGS = ("", "a", "owi", "em", "iem", "u", "ie", "e", "y", "ów", "om", "ami", "ach")


def _forms():
    out = set()
    for n in FEMALE + MALE:
        n = n.lower()
        out.add(n)
        if n.endswith("a"):
            out.update(n[:-1] + e for e in _A_ENDINGS)
        elif n.endswith("i") or n.endswith("y"):
            out.update(n + e for e in ("ego", "emu", "m", "ch"))
        else:
            out.update(n + e for e in _C_ENDINGS)
    return out


NAME_FORMS = _forms()


def is_polish_name(word: str) -> bool:
    return word.lower() in NAME_FORMS
