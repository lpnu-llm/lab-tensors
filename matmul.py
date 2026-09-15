# %% [markdown]

# # Операції з тензорами
#
# Нейронні мережі будуються на матричних операціях, насамперед на матричному
# добутку. Щоб зробити трансформер з нуля, нам постійно доведеться працювати з
# векторами, матрицями та тензорами.
#
# Матричний добуток — це фундаментальна операція. Саме на неї припадатиме більша
# частина часу та обчислювальних ресурсів при використанні великих мовних
# моделей.

# Оскільки ця операція така важлива, є багато способів, як пришвидшити тензорний
# добуток. Про це ми поговоримо в окремому курсі (LLM Systems).
#
# Крім того, існує багато бібліотек для роботи з тензорами. У світі deep
# learning та великих мовних моделей найпопулярнішою є PyTorch, яка в свою чергу
# сильно надихалася API NumPy.

# У реальному житті ви напевно будете використовувати саме PyTorch, але зараз ми
# імплементуємо всі операції з нуля без використання сторонніх бібліотек.

# Трохи термінології:
# - тензор - багатовимірний числовий масив
# - розмірність, ранг, порядок тензора - кількість осей
# - скаляр - одне число, яке можна вважати 0-вимірним тензором
# - вектор - 1-вимірний тензор
# - матриця - 2-вимірний тензор


# %%

def vector_add(a, b):
    """Додає два вектори поелементно."""

    ...

# Питання:
# - Скільки числових додавань використовується для векторів довжини n?
# - Скільки додаткової пам'яті потрібно для результату?


def test_vector_add():
    assert vector_add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert vector_add([0.5, 1.5], [0.25, -0.5]) == [0.75, 1.0]
    assert vector_add([], []) == []

    try:
        vector_add([1, 2], [3, 4, 5])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Додавання векторів різних довжин має давати помилку")


if __name__ == "__main__":
    test_vector_add()
    print("✓ vector_add")

# %%

def dot(a, b):
    """Скалярний добуток двох векторів."""
    ...

# Питання:
# - Скільки операцій множення та додавання виконується для векторів довжиною n?
# - Якщо подвоїти довжину векторів, як зміниться кількість операцій?

def test_dot():
    assert dot([1, 2, 3], [4, 5, 6]) == 32
    assert dot([2, -3], [4, 0.5]) == 6.5
    assert dot([], []) == 0

    # Ненульові вектори теж можуть мати нульовий скалярний добуток.
    assert dot([1, 0], [0, 1]) == 0

    try:
        dot([1, 2], [3])
    except ValueError:
        pass
    else:
        raise AssertionError("Добуток векторів різних неможливий")


if __name__ == "__main__":
    test_dot()
    print("✓ dot")

# %% [markdown]
#
# ## Множення матриці на вектор

# Множення матриці не вектор це як `dot` для багатьох векторів одразу
#
# Скалярний добуток двох векторів `dot(a, b)` дає одне число. Але часто буває
# так, що у нас є один вхідний вектор `x`, і його треба перемножити не з одним,
# а з багатьма різними векторами $a_1, a_2, \dots, a_m$.
#
# Один варіант це написати цикл і викликати `dot` $m$ разів. А можна зробити
# інакше. Складемо всі ці вектори рядками в одну матрицю
#
# $$ A = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_m \end{bmatrix} $$
#
# і виконаємо одну операцію, множення матриці на вектор `matvec(A, x)`.
# Результатом буде вектор з $m$ чисел. Перше з них дорівнює `dot(a_1, x)`, друге
# дорівнює `dot(a_2, x)`, і так далі.
#
# Тобто множення матриці на вектор є просто `dot`, застосованим до кожного
# рядка.
#
# ### Приклад: кошик покупок
#
# Ви купуєте 2 буханки хліба, 1 кг яблук і 3 л молока. Запишемо це як вектор.
#
# ```python
# vec = [2, 1, 3]  # хліб, яблука, молоко
# ```
#
# Ціни на ці товари є у двох магазинах. Кожен рядок матриці є прайсом одного
# магазину, записаним у тому ж порядку (хліб, яблука, молоко).
#
# ```python
# mat = [
#     [1, 2, 3],  # магазин A: хліб=1, яблука=2, молоко=3
#     [4, 5, 6],  # магазин B: хліб=4, яблука=5, молоко=6
# ]
# ```
#
# Скільки ви заплатите в кожному магазині? Для одного магазину відповідь є `dot`
# вашого кошика з його прайсом:
#
# $$
# \begin{aligned}
# \text{магазин A:}\quad & 1 \cdot 2 + 2 \cdot 1 + 3 \cdot 3 = 13 \\
# \text{магазин B:}\quad & 4 \cdot 2 + 5 \cdot 1 + 6 \cdot 3 = 31
# \end{aligned}
# $$
#
# А для всіх магазинів одразу відповідь дає множення матриці на вектор:
#
# ```python
# matvec(mat, vec) == [13, 31]
# ```
#
# Той самий результат у вигляді вектора.
#
# ### Як про це думати
#
# Вектор є вхідними даними (наш кошик). Кожен рядок матриці є окремим
# «рецептом», як стиснути цей вхід в одне число (прайс конкретного магазину).
# Результат є вектором, у якому по одному числу на кожен рядок (сума чеку в
# кожному магазині).
#
# Звідси одразу випливають вимоги до розмірів.
#
# Рядок матриці має бути такої ж довжини, як вектор, інакше не буде з чим
# перемножувати. Не можна помножити ціну молока на кількість товару, якого немає
# в кошику.
#
# Кількість рядків матриці задає довжину результату. Скільки «рецептів», стільки
# й чисел на виході.
#
# mat форми (m, n) * vec довжини n -> результат довжини m
#
# У нашому прикладі (2, 3) * 3 -> 2. Три товари, два магазини, дві суми.
#
# ### Формально
#
# $$ y_i = \sum_{j=0}^{n-1} A_{ij}\, x_j = \operatorname{dot}(A_{i,:},\ x) $$
#
# де $A_{i,:}$ позначає весь рядок з індексом $i$. Це і є запис словами «для
# кожного рядка $i$ порахувати `dot` цього рядка з вектором $x$».

# %%

def matvec(mat, vec):
    """Матрично-векторний добуток."""

    # Використати dot
    ...


def test_matvec():
    assert matvec(
        [[1, 2, 3],
         [4, 5, 6]],
        [10, 20, 30],
    ) == [140, 320]

    # Одинична матриця не змінює вектор.
    assert matvec(
        [[1, 0],
         [0, 1]],
        [7, -3],
    ) == [7, -3]

    # Один рядок: результат усе одно вектор, а не скаляр.
    assert matvec([[2, 3]], [4, 5]) == [23]

    # Один стовпець.
    assert matvec([[2], [3], [4]], [10]) == [20, 30, 40]

    assert matvec([[0, 0], [0, 0]], [5, 6]) == [0, 0]
    assert matvec([[], []], []) == [0.0, 0.0]

    a = [[1, 2], [3, 4]]
    x = [5, 6]
    matvec(a, x)
    assert a == [[1, 2], [3, 4]]
    assert x == [5, 6]

    # Рядки матриці мають мати ту саму довжину, що й вектор.
    try:
        matvec([
            [1, 2],
            [3, 4]],
        [1])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Очікується ValueError: довжина вектора не відповідає довжині рядків"
        )

    # Усі рядки матриці мають бути однакової довжини.
    try:
        matvec([
            [1, 2],
            [3]],
        [1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Очікується ValueError: матриця має рядки різної довжини"
        )

    # Для порожньої матриці неможливо визначити кількість її стовпців.
    try:
        matvec([], [])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Очікується ValueError: порожня матриця не визначає кількість стовпців"
        )

    print("Усі перевірки matvec пройдено.")


if __name__ == "__main__":
    test_matvec()
    print("✓ matvec")

# %% [markdown]
#
# ## Форма тензора
#
# Перед тим як поєднувати тензори в операціях, важливо вміти визначати їхню
# **форму** (shape). Форма показує розмір тензора вздовж кожної його осі.
#
# - скаляр не має осей, тому його форма — `()`;
# - вектор з $n$ чисел має форму `(n,)`;
# - матриця з $m$ рядків і $n$ стовпців має форму `(m, n)`;
# - тривимірний тензор може мати форму `(d1, d2, d3)`.
#
# ```python
# shape(7) == ()
# shape([10, 20, 30]) == (3,)
# shape([[1, 2, 3],
#        [4, 5, 6]]) == (2, 3)
# ```
#
# Форма існує лише тоді, коли всі вкладені списки мають однакову форму.
# Наприклад, `[[1, 2], [3]]` не є матрицею, бо її рядки мають різну довжину.

# %%

def shape(tensor):
    """Повертає форму тензора як tuple."""

    # Для вкладених списків цю задачу зручно розв'язати рекурсивно.
    ...


def test_shape():
    assert shape(7) == ()
    assert shape(3.14) == ()

    assert shape([10, 20, 30]) == (3,)
    assert shape([]) == (0,)

    assert shape(
        [[1, 2, 3],
         [4, 5, 6]],
    ) == (2, 3)
    assert shape([[1], [2], [3]]) == (3, 1)
    assert shape([[], []]) == (2, 0)

    assert shape(
        [[[1, 2], [3, 4]],
         [[5, 6], [7, 8]]],
    ) == (2, 2, 2)

    # Тензор не може мати вкладені списки різної форми.
    try:
        shape([[1, 2], [3]])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Очікується ValueError: вкладені списки мають різну форму"
        )

    try:
        shape([1, [2]])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Очікується ValueError: тензор змішує скаляри та списки"
        )

    print("Усі перевірки shape пройдено.")


if __name__ == "__main__":
    test_shape()
    print("✓ shape")

# %% [markdown]
#
# ## Лінійний шар
#
# Тепер застосуємо `matvec` так, як це роблять у нейронних мережах.
# Лінійний шар перетворює вхідний вектор $x$ на вихідний вектор $y$:
#
# $$ y = Wx + b $$
#
# Тут:
#
# - `input_dim` — кількість чисел у вхідному векторі `x`;
# - `out_dim` — кількість чисел у вихідному векторі `y`;
# - `W` — матриця ваг форми `(out_dim, input_dim)`;
# - `b` — вектор зсуву (bias) довжини `out_dim`.
#
# Це можна перевірити за допомогою `shape`: `shape(x) == (input_dim,)`,
# `shape(W) == (out_dim, input_dim)` і `shape(b) == (out_dim,)`.
#
# Спочатку `matvec(W, x)` створює вектор довжини `out_dim`. Потім до
# нього поелементно додається `b`. Отже, цей шар може змінювати
# розмір вектора:
#
# $$
# \underbrace{x}_{\text{input_dim}}
# \quad\longrightarrow\quad
# \underbrace{y}_{\text{out_dim}}
# $$
#
# Наприклад, шар з `input_dim = 2` і `out_dim = 3` збільшує вектор
# з двох чисел до трьох:
#
# ```python
# x = [2, 3]              # input_dim = 2
# W = [[1, 0],
#      [0, 1],
#      [1, 1]]            # shape = (out_dim, input_dim) = (3, 2)
# b = [10, 20, 30]        # out_dim = 3
#
# linear(x, W, b) == [12, 23, 35]
# ```
#
# Якщо `out_dim` менший за `input_dim`, шар, навпаки, стискає вектор.

# %%

def linear(x, weight, bias):
    """Застосовує лінійний шар: y = Wx + b."""

    # Використайте минулі функції.
    ...


def test_linear():
    # input_dim = 2, out_dim = 3: шар збільшує розмір вектора.
    assert linear(
        [2, 3],
        [[1, 0],
         [0, 1],
         [1, 1]],
        [10, 20, 30],
    ) == [12, 23, 35]

    # input_dim = 3, out_dim = 1: шар стискає вектор.
    assert linear(
        [2, 3, 4],
        [[1, 10, 100]],
        [5],
    ) == [437]

    # Нульові ваги: результат дорівнює bias.
    assert linear(
        [7, -3],
        [[0, 0],
         [0, 0]],
        [1, -2],
    ) == [1, -2]

    # Довжина bias має дорівнювати out_dim.
    try:
        linear(
            [1, 2],
            [[1, 0],
             [0, 1]],
            [10],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Очікується ValueError: довжина bias не дорівнює out_dim"
        )

    print("Усі перевірки linear пройдено.")


if __name__ == "__main__":
    test_linear()
    print("✓ linear")

# %% [markdown]
#
# ## Множення матриць
#
# `matvec` множить матрицю на один вектор. Але що робити, якщо векторів
# багато? Їх можна скласти у матрицю і виконати множення матриць.
#
# Нехай `A` має форму `(m, n)`, а `B` — форму `(n, p)`. Їхній добуток
# `C = AB` має форму `(m, p)`:
#
# $$
# \underbrace{A}_{m \times n}
# \underbrace{B}_{n \times p}
# =
# \underbrace{C}_{m \times p}
# $$
#
# Внутрішні розміри `n` мають збігатися. Кожен елемент результату — це
# скалярний добуток рядка `A` і стовпця `B`:
#
# $$ C_{ij} = \sum_{k=0}^{n-1} A_{ik}B_{kj} $$
#
# Функція `shape` допомагає зрозуміти, чи можна перемножити дві матриці. Якщо:
#
# ```python
# a_shape = shape(A)  # (m, n)
# b_shape = shape(B)  # (n, p)
# ```
#
# то множення можливе, коли `a_shape[1] == b_shape[0]`. Тобто кількість стовпців
# першої матриці має дорівнювати кількості рядків другої. Форма результату
# буде `(a_shape[0], b_shape[1])`, тобто `(m, p)`.
#
# Наприклад, `(2, 3) @ (3, 4)` можна обчислити, і результат матиме форму `(2, 4)`.
# А `(2, 3) @ (2, 4)` обчислити неможливо, бо `3 != 2`.
#
# Наприклад:
#
# ```python
# A = [[1, 2, 3],
#      [4, 5, 6]]          # shape = (2, 3)
#
# B = [[1, 2],
#      [3, 4],
#      [5, 6]]             # shape = (3, 2)
#
# matmul(A, B) == [[22, 28],
#                  [49, 64]]  # shape = (2, 2)
# ```

# %%

def matmul(a, b):
    """Множить матрицю a форми (m, n) на матрицю b форми (n, p)."""

    # 1. Знайдіть форми a і b за допомогою shape та перевірте їх.
    # 2. Використайте dot для кожної пари: рядок a та стовпець b.
    ...


def test_matmul():
    assert matmul(
        [[1, 2, 3],
         [4, 5, 6]],
        [[1, 2],
         [3, 4],
         [5, 6]],
    ) == [[22, 28], [49, 64]]

    # Прямокутні матриці: (3, 2) @ (2, 4) -> (3, 4).
    assert matmul(
        [[1, 2],
         [3, 4],
         [5, 6]],
        [[1, 0, 1, 0],
         [0, 1, 0, 1]],
    ) == [[1, 2, 1, 2],
          [3, 4, 3, 4],
          [5, 6, 5, 6]]

    # Множення на одиничну матрицю не змінює матрицю.
    assert matmul(
        [[2, -1],
         [7, 3]],
        [[1, 0],
         [0, 1]],
    ) == [[2, -1], [7, 3]]

    # Внутрішні розміри мають збігатися.
    try:
        matmul(
            [[1, 2, 3]],
            [[1, 2],
             [3, 4]],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Очікується ValueError: внутрішні розміри матриць не збігаються"
        )

    # Усі рядки кожної матриці мають бути однакової довжини.
    try:
        matmul(
            [[1, 2],
             [3]],
            [[1],
             [2]],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Очікується ValueError: матриця a має рядки різної довжини"
        )

    print("Усі перевірки matmul пройдено.")


if __name__ == "__main__":
    test_matmul()
    print("✓ matmul")

# %% [markdown]
#
# ## Транспонування матриці
#
# Транспонування міняє рядки та стовпці матриці місцями. Транспоновану
# матрицю `A` позначають як $A^T$.
#
# Якщо `A` має форму `(m, n)`, то $A^T$ має форму `(n, m)`:
#
# $$
# \underbrace{A}_{m \times n}
# \quad\longrightarrow\quad
# \underbrace{A^T}_{n \times m}
# $$
#
# Елемент з рядка `i` і стовпця `j` переходить у рядок `j` і стовпець `i`:
#
# $$ (A^T)_{ji} = A_{ij} $$
#
# ```python
# A = [[1, 2, 3],
#      [4, 5, 6]]          # shape = (2, 3)
#
# transpose(A) == [[1, 4],
#                  [2, 5],
#                  [3, 6]]  # shape = (3, 2)
# ```
#
# Кожен рядок нової матриці — це один стовпець початкової матриці.

# %%

def transpose(mat):
    """Міняє рядки та стовпці матриці місцями."""

    ...


def test_transpose():
    assert transpose(
        [[1, 2, 3],
         [4, 5, 6]],
    ) == [[1, 4],
          [2, 5],
          [3, 6]]

    # Рядок стає стовпцем.
    assert transpose([[1, 2, 3]]) == [[1], [2], [3]]

    # Стовпець стає рядком.
    assert transpose([[1], [2], [3]]) == [[1, 2, 3]]

    # Квадратна матриця теж змінюється, якщо вона несиметрична.
    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]

    # Усі рядки матриці мають бути однакової довжини.
    try:
        transpose([[1, 2], [3]])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Очікується ValueError: матриця має рядки різної довжини"
        )

    print("Усі перевірки transpose пройдено.")


if __name__ == "__main__":
    test_transpose()
    print("✓ transpose")

# %% [markdown]
#
# ## Батчинг у нейронних мережах
#
# Під час реального застосування нейронні мережі часто виконують інференс для кількох вибірок
# з даних одночасно. Таку групу прикладів називають **батчем** (batch). GPU може
# обробити багато прикладів паралельно, тому батчинг зазвичай ефективніший,
# ніж обробка кожного прикладу окремо.
#
# Запишемо кожен вхідний вектор як рядок матриці `X`:
#
# - `X` має форму `(batch_size, input_dim)`;
# - `W` має форму `(out_dim, input_dim)`;
# - `W_T`, транспонована `W`, має форму `(input_dim, out_dim)`;
# - `XW_T` має форму `(batch_size, out_dim)`.
#
# Використаємо щойно створену функцію `transpose(W)`, щоб внутрішні розміри
# для `matmul(X, W_T)` збігалися.
#
# ```python
# X = [[2, 3],             # перший приклад
#      [4, 5],             # другий приклад
#      [6, 7]]             # третій приклад
#                         # shape = (batch_size, input_dim) = (3, 2)
#
# W = [[1, 0],
#      [0, 1],
#      [1, 1]]            # shape = (out_dim, input_dim) = (3, 2)
#
# W_T = [[1, 0, 1],
#        [0, 1, 1]]       # shape = (input_dim, out_dim) = (2, 3)
#
# matmul(X, W_T) == [[2, 3, 5],
#                    [4, 5, 9],
#                    [6, 7, 13]]
#                         # shape = (batch_size, out_dim) = (3, 3)
# ```
#
# Так одне множення матриць замінює `matvec(W, x)` для кожного прикладу у батчі.
# Після цього вектор `b` додається до кожного рядка результату.

# %%

def linear_batch(xs, weight, bias):
    """Застосовує лінійний шар до батчу вхідних векторів."""

    # 1. Транспонуйте weight за допомогою transpose.
    # 2. Виконайте matmul(xs, weight_transposed).
    # 3. Додайте bias до кожного рядка за допомогою vector_add.
    ...


def test_linear_batch():
    xs = [[2, 3],
          [4, 5],
          [6, 7]]
    weight = [[1, 0],
              [0, 1],
              [1, 1]]
    bias = [10, 20, 30]

    result = linear_batch(xs, weight, bias)

    assert result == [[12, 23, 35],
                      [14, 25, 39],
                      [16, 27, 43]]

    # Обробка батчу має давати ті самі результати, що й обробка по одному.
    assert result == [linear(x, weight, bias) for x in xs]

    print("Усі перевірки linear_batch пройдено.")


if __name__ == "__main__":
    test_linear_batch()
    print("✓ linear_batch")

# %% [markdown]
#
# ## Порівняння: цикл з `matvec` чи один `matmul`
#
# Ми вже знаємо два способи застосувати одні й ті самі ваги до цілого батчу:
#
# 1. пройти циклом по вхідних векторах і викликати `matvec(W, x)` для кожного;
# 2. скласти всі входи у матрицю `X` і один раз викликати `matmul(X, transpose(W))`.
#
# Обидва способи мають дати однаковий результат. Функція нижче спочатку це
# перевіряє, а потім вимірює час обох варіантів.
#
# У нашій навчальній реалізації обидві функції всередині використовують цикли Python,
# тому `matmul` може не виявитися швидшим у цьому експерименті. У PyTorch, NumPy та на GPU
# матричне множення реалізоване оптимізованими низькорівневими програмами. Вони можуть
# використати паралелізм апаратного забезпечення, тому один великий `matmul` зазвичай
# ефективніший за багато малих викликів `matvec`.

# %%

def compare_matvec_and_matmul():
    """Порівнює час обробки батчу двома способами."""
    from time import perf_counter

    batch_size = 32
    input_dim = 64
    out_dim = 48
    repeats = 5

    # Створюємо входи та ваги без сторонніх бібліотек.
    xs = [
        [(sample + feature) % 10 for feature in range(input_dim)]
        for sample in range(batch_size)
    ]
    weight = [
        [(output - feature) % 7 for feature in range(input_dim)]
        for output in range(out_dim)
    ]

    # Транспонуємо ваги один раз, як це можна зробити перед інференсом.
    weight_transposed = transpose(weight)

    by_matvec = [matvec(weight, x) for x in xs]
    by_matmul = matmul(xs, weight_transposed)
    assert by_matvec == by_matmul

    start = perf_counter()
    for _ in range(repeats):
        [matvec(weight, x) for x in xs]
    matvec_seconds = perf_counter() - start

    start = perf_counter()
    for _ in range(repeats):
        matmul(xs, weight_transposed)
    matmul_seconds = perf_counter() - start

    print(f"Цикл з matvec: {matvec_seconds * 1000:.2f} мс")
    print(f"Один matmul:    {matmul_seconds * 1000:.2f} мс")
    print(f"Відношення:     {matvec_seconds / matmul_seconds:.2f}x")


if __name__ == "__main__":
    compare_matvec_and_matmul()

# %% [markdown]
#
# ## Порівняння з PyTorch: Python, CPU та GPU
#
# Наша функція `matmul` допомогла зрозуміти матричне множення, але вона
# виконує мільйони операцій у циклах Python. PyTorch передає цю роботу
# оптимізованим програмам для CPU або GPU.
#
# Остання комірка порівнює:
#
# - наш `matmul` на Python;
# - `torch.matmul` на CPU;
# - `torch.matmul` на NVIDIA GPU через CUDA або на Apple GPU через MPS.
#
# Для маленьких матриць GPU може не бути швидшим за CPU. Запуск операції на GPU
# має власні накладні витрати, і GPu розрахований на сотні-тисячі маленьких операцій одночасно.
# Але маленькі матриці, такі як 3*3, не можуть в собі мати таку кількість обчислень, що робить GPU
# нерелевантним. На великих матрицях паралелізм GPU зазвичай дає значну перевагу.
#
# Ми створюємо тензори на потрібному пристрої **до** початку вимірювання. Так ми
# вимірюємо час обчислення, а не копіювання даних між CPU і GPU. У реальній програмі
# це копіювання теж може впливати на загальний час.

# %%

def compare_with_torch():
    """Порівнює навчальний matmul з PyTorch на CPU та GPU."""
    from time import perf_counter

    try:
        import torch
    except ImportError:
        print("PyTorch не встановлено. Встановіть його, щоб запусти цей бенчмарк.")
        return

    def synchronize(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def measure_torch(a, b, repeats):
        # Перші запуски можуть мати додаткові витрати на підготовку.
        for _ in range(3):
            torch.matmul(a, b)
        synchronize(a.device)

        start = perf_counter()
        for _ in range(repeats):
            torch.matmul(a, b)
        synchronize(a.device)
        return (perf_counter() - start) / repeats

    def make_python_matrix(size, offset):
        return [
            [float((row + column + offset) % 7) for column in range(size)]
            for row in range(size)
        ]

    def benchmark_size(size, python_repeats, torch_repeats):
        a = make_python_matrix(size, 0)
        b = make_python_matrix(size, 1)

        start = perf_counter()
        python_result = None
        for _ in range(python_repeats):
            python_result = matmul(a, b)
        python_time = (perf_counter() - start) / python_repeats

        a_cpu = torch.tensor(a, dtype=torch.float32)
        b_cpu = torch.tensor(b, dtype=torch.float32)
        cpu_time = measure_torch(a_cpu, b_cpu, torch_repeats)

        # Перевіряємо, що оптимізована функція дає той самий результат.
        expected = torch.tensor(python_result, dtype=torch.float32)
        assert torch.allclose(torch.matmul(a_cpu, b_cpu), expected)

        return python_time, cpu_time, a_cpu, b_cpu

    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_device = torch.device("mps")
    else:
        gpu_device = None

    print(f"PyTorch {torch.__version__}")
    print("\nНаш matmul і PyTorch CPU:")
    print(f"{'size':>10} {'Python':>14} {'PyTorch CPU':>14} {'CPU speedup':>14}")

    # 16x16 — мала матриця; 128x128 — вже мільйони Python-операцій.
    small = benchmark_size(16, python_repeats=20, torch_repeats=200)
    large = benchmark_size(128, python_repeats=1, torch_repeats=30)

    for size, result in ((16, small), (128, large)):
        python_time, cpu_time, _, _ = result
        print(
            f"{f'{size}x{size}':>10} "
            f"{python_time * 1000:>11.3f} мс "
            f"{cpu_time * 1000:>11.3f} мс "
            f"{python_time / cpu_time:>13.1f}x"
        )

    if gpu_device is None:
        print("\nCUDA або MPS недоступні: GPU-бенчмарк пропущено.")
        return

    print(f"\nPyTorch CPU і {gpu_device.type.upper()} GPU:")
    print(f"{'size':>10} {'PyTorch CPU':>14} {gpu_device.type.upper() + ' GPU':>14} {'GPU speedup':>14}")

    # Малі матриці з попереднього бенчмарку.
    _, small_cpu_time, small_a_cpu, small_b_cpu = small
    small_a_gpu = small_a_cpu.to(gpu_device)
    small_b_gpu = small_b_cpu.to(gpu_device)
    small_gpu_time = measure_torch(small_a_gpu, small_b_gpu, repeats=200)
    print(
        f"{'16x16':>10} "
        f"{small_cpu_time * 1000:>11.3f} мс "
        f"{small_gpu_time * 1000:>11.3f} мс "
        f"{small_cpu_time / small_gpu_time:>13.2f}x"
    )

    # Більші матриці, які ми також запускали через наш Python matmul.
    _, large_cpu_time, large_a_cpu, large_b_cpu = large
    large_a_gpu = large_a_cpu.to(gpu_device)
    large_b_gpu = large_b_cpu.to(gpu_device)
    large_gpu_time = measure_torch(large_a_gpu, large_b_gpu, repeats=30)
    print(
        f"{'128x128':>10} "
        f"{large_cpu_time * 1000:>11.3f} мс "
        f"{large_gpu_time * 1000:>11.3f} мс "
        f"{large_cpu_time / large_gpu_time:>13.2f}x"
    )

    # Дуже великий розмір запускаємо лише в PyTorch: цикли Python були б надто повільними.
    huge_size = 1024
    huge_a_cpu = torch.randn(huge_size, huge_size)
    huge_b_cpu = torch.randn(huge_size, huge_size)
    huge_cpu_time = measure_torch(huge_a_cpu, huge_b_cpu, repeats=10)
    huge_a_gpu = huge_a_cpu.to(gpu_device)
    huge_b_gpu = huge_b_cpu.to(gpu_device)
    huge_gpu_time = measure_torch(huge_a_gpu, huge_b_gpu, repeats=10)
    print(
        f"{'1024x1024':>10} "
        f"{huge_cpu_time * 1000:>11.3f} мс "
        f"{huge_gpu_time * 1000:>11.3f} мс "
        f"{huge_cpu_time / huge_gpu_time:>13.2f}x"
    )

    print("\nЗначення можуть відрізнятися залежно від комп'ютера та його поточного навантаження.")


if __name__ == "__main__":
    compare_with_torch()
