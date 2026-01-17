import logging
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("statistical_report.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

excel_path = r"data/data.xls"
sheet_index = 22
use_col = 0

try:
    col = pd.read_excel(excel_path, sheet_name=sheet_index, usecols=[use_col])
    series = pd.to_numeric(col.iloc[:, 0], errors="coerce").dropna().astype(float)
    data = series.tolist()
except Exception as e:
    logger.error(f"Ошибка чтения файла: {e}")
    sys.exit()

n = len(data)
logger.info("=" * 60)
logger.info(f"СТАТИСТИЧЕСКИЙ ОТЧЕТ ПО ВЫБОРКЕ (n = {n})")
logger.info("=" * 60)

# --- 7. Первичный анализ ---
sum_x = sum(data)
sum_x2 = sum(x**2 for x in data)
mean_x = sum_x / n
mean_x2 = sum_x2 / n
var_x = mean_x2 - (mean_x**2)
std_x = math.sqrt(var_x)

sorted_data = sorted(data)
if n % 2 == 1:
    median = sorted_data[n // 2]
else:
    median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2

m3 = sum((x - mean_x) ** 3 for x in data) / n
m4 = sum((x - mean_x) ** 4 for x in data) / n
As = m3 / (std_x**3) if std_x != 0 else 0
Ex = (m4 / (std_x**4) - 3) if std_x != 0 else 0

logger.info("\n[ЗАДАНИЕ 7] ВЫБОРОЧНЫЕ ХАРАКТЕРИСТИКИ")
logger.info(f"1. Сумма X = {sum_x:.4f}")
logger.info(f"2. Среднее (X_cp) = ΣX / n = {sum_x:.4f} / {n} = {mean_x:.4f}")
logger.info(f"3. Сумма X^2 = {sum_x2:.4f}")
logger.info(f"4. Среднее квадратов (X^2_cp) = ΣX^2 / n = {mean_x2:.4f}")
logger.info(
    f"5. Дисперсия (S^2) = X^2_cp - (X_cp)^2 = {mean_x2:.4f} - ({mean_x:.4f})^2 = {var_x:.4f}"
)
logger.info(f"6. Ср.кв.отклонение (S) = √S^2 = {std_x:.4f}")
logger.info(f"7. Медиана (x_med): {median:.4f}")
logger.info(f"8. Асимметрия (As = m3 / S^3): {As:.4f}")
logger.info(f"9. Эксцесс (Ex = m4 / S^4 - 3): {Ex:.4f}")

# --- 8. Критерий серий ---
signs = []
for x in data:
    if x > median:
        signs.append("+")
    elif x < median:
        signs.append("-")

n1 = signs.count("+")
n2 = signs.count("-")
ks = 1
for i in range(1, len(signs)):
    if signs[i] != signs[i - 1]:
        ks += 1

mean_ks = (2 * n1 * n2) / (n1 + n2) + 1
var_ks = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
std_ks = math.sqrt(var_ks)
z_calc_series = (
    (ks - mean_ks - 0.5) / std_ks if ks < mean_ks else (ks - mean_ks + 0.5) / std_ks
)

logger.info("\n[ЗАДАНИЕ 8] ПРОВЕРКА СЛУЧАЙНОСТИ (Критерий серий)")
logger.info(f"1. Число элементов > медианы (n1) = {n1}")
logger.info(f"2. Число элементов < медианы (n2) = {n2}")
logger.info(f"3. Количество серий (KS) = {ks}")
logger.info(f"4. M[KS] = (2*n1*n2)/(n1+n2) + 1 = {mean_ks:.4f}")
logger.info(
    f"5. σ[KS] = √[ (2n1n2(2n1n2-n1-n2)) / ((n1+n2)^2 * (n1+n2-1)) ] = {std_ks:.4f}"
)
logger.info(f"6. Z_выч = (KS - M[KS] ± 0.5) / σ[KS] = {z_calc_series:.4f}")
logger.info(f"7. Критическое Z_крит (α=0.05) = 1.96")
logger.info(
    f"ВЫВОД: {'Случайна' if abs(z_calc_series) < 1.96 else 'Не случайна'} (т.к. |Z| {'<' if abs(z_calc_series) < 1.96 else '>'} 1.96)"
)

# --- 9. Оценка максимального правдоподобия (ОМП) для F4 ---
logger.info("\n" + "=" * 60)
logger.info("[ЗАДАНИЕ 9] ОЦЕНКА МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ (ОМП)")
logger.info("=" * 60)
logger.info("\nГипотетическое распределение F4:")
logger.info("  Плотность: f(x; θ) = 2x / θ²,  при 0 ≤ x ≤ θ")
logger.info("  Неизвестный параметр: θ > 0")

logger.info("\n--- Вывод ОМП для параметра θ ---")
logger.info("\n1. Функция правдоподобия для выборки x₁, x₂, ..., xₙ:")
logger.info("   L(θ) = ∏ᵢ f(xᵢ; θ) = ∏ᵢ (2xᵢ / θ²)")
logger.info("   L(θ) = (2ⁿ · ∏ᵢxᵢ) / θ^(2n)")
logger.info("   при условии: θ ≥ max(xᵢ) (иначе L(θ) = 0)")

logger.info("\n2. Логарифм функции правдоподобия:")
logger.info("   ln L(θ) = n·ln(2) + Σᵢln(xᵢ) - 2n·ln(θ)")

logger.info("\n3. Производная по θ:")
logger.info("   d[ln L(θ)]/dθ = -2n / θ")
logger.info("   Производная < 0 для всех θ > 0")
logger.info("   => ln L(θ) монотонно убывает по θ")

logger.info("\n4. Анализ максимума:")
logger.info("   - L(θ) = 0 при θ < max(xᵢ) (плотность не определена)")
logger.info("   - L(θ) убывает при θ ≥ max(xᵢ)")
logger.info("   - Максимум достигается при минимальном допустимом θ")

logger.info("\n5. ВЫВОД: ОМП для θ:")
logger.info("   θ̂ = max(x₁, x₂, ..., xₙ) = X₍ₙ₎")

# Вычисление ОМП
theta_mle = max(data)
x_min = min(data)
prod_x = math.exp(
    sum(math.log(x) for x in data) / n
)  # Геометрическое среднее для отображения

logger.info("\n--- Численный расчет ---")
logger.info(f"   Минимальное значение выборки: min(X) = {x_min:.4f}")
logger.info(f"   Максимальное значение выборки: max(X) = {theta_mle:.4f}")
logger.info(f"\n   ═══════════════════════════════════════")
logger.info(f"   ОМП параметра θ:  θ̂ = {theta_mle:.4f}")
logger.info(f"   ═══════════════════════════════════════")

# Проверка: значение функции правдоподобия в точке ОМП
log_likelihood = (
    n * math.log(2) + sum(math.log(x) for x in data) - 2 * n * math.log(theta_mle)
)
logger.info(f"\n   Проверка: ln L(θ̂) = {log_likelihood:.4f}")

theta = theta_mle  # Используем ОМП для дальнейших расчетов

# --- 10. Хи-квадрат ---
k_sturges = int(math.log2(n)) + 1
step = theta / k_sturges
chi_sq = 0
logger.info(f"\n[ЗАДАНИЕ 10] КРИТЕРИЙ ХИ-КВАДРАТ (α=0.1, Распределение F4)")
logger.info(f"Используется ОМП: θ̂ = {theta:.4f}")
logger.info(f"Формула плотности: f(x) = 2x / {theta:.4f}²")
logger.info(f"Число интервалов (по Стёрджесу): k = [log₂({n})] + 1 = {k_sturges}")
logger.info(
    f"\n{'Интервал':<20} | {'n_выб':<6} | {'p_i':<8} | {'n_теор':<8} | {'(n-n_т)²/n_т':<12}"
)
logger.info("-" * 75)

for i in range(k_sturges):
    left = i * step
    right = (i + 1) * step
    if i < k_sturges - 1:
        ni_obs = len([x for x in data if left <= x < right])
    else:
        ni_obs = len([x for x in data if left <= x <= right])

    p_i = (right**2 - left**2) / (theta**2)
    ni_theor = n * p_i
    term = ((ni_obs - ni_theor) ** 2) / ni_theor if ni_theor > 0 else 0
    chi_sq += term

    logger.info(
        f"[{left:5.2f}; {right:5.2f}) | {ni_obs:6d} | {p_i:8.4f} | {ni_theor:8.2f} | {term:12.4f}"
    )

nu = k_sturges - 1 - 1
logger.info("-" * 75)
logger.info(f"1. Хи-квадрат выч = Σ[...] = {chi_sq:.4f}")
logger.info(f"2. Степени свободы ν = k - r - 1 = {k_sturges} - 1 - 1 = {nu}")
logger.info(f"3. Для α=0.1 и ν={nu} найди χ²_крит в таблице 5")

# --- 11. График гистограммы и плотности ---
logger.info("\n" + "=" * 60)
logger.info("[ЗАДАНИЕ 11] ГРАФИК ПЛОТНОСТИ И ГИСТОГРАММЫ")
logger.info("=" * 60)
logger.info(f"\nПараметры построения:")
logger.info(f"  - Число интервалов гистограммы: k = {k_sturges}")
logger.info(f"  - ОМП параметра θ: θ̂ = {theta:.4f}")
logger.info(f"  - Плотность: f(x) = 2x / θ̂² = 2x / {theta**2:.4f}")

try:
    fig, ax = plt.subplots(figsize=(12, 7))

    # Гистограмма (нормированная по плотности)
    counts, bins, patches = ax.hist(
        data,
        bins=k_sturges,
        density=True,
        edgecolor="black",
        color="skyblue",
        alpha=0.7,
        label="Гистограмма (эмпирическая плотность)",
    )

    # Теоретическая плотность F4 с ОМП
    x_vals = np.linspace(0, theta, 500)
    f_x = (2 * x_vals) / (theta**2)
    ax.plot(
        x_vals,
        f_x,
        "r-",
        lw=2.5,
        label=rf"Плотность $F_4$: $f(x) = \frac{{2x}}{{\hat{{\theta}}^2}}$, $\hat{{\theta}}={theta:.4f}$",
    )

    # Отметка ОМП на графике
    ax.axvline(
        x=theta,
        color="green",
        linestyle="--",
        lw=1.5,
        alpha=0.8,
        label=rf"ОМП: $\hat{{\theta}} = {theta:.4f}$",
    )

    # Настройки графика
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Плотность вероятности f(x)", fontsize=12)
    ax.set_title(
        r"Гистограмма выборки и теоретическая плотность $F_4$ с ОМП параметра $\theta$",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, theta * 1.05)
    ax.set_ylim(0, max(max(counts) * 1.1, max(f_x) * 1.1))

    # Добавляем текстовую аннотацию
    textstr = f"n = {n}\nОМП: θ̂ = {theta:.4f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    fig.savefig("histogram.png", dpi=150)
    plt.close(fig)
    logger.info("\n-> График успешно сохранен в `histogram.png`")
    logger.info("   На графике отображены:")
    logger.info("   - Гистограмма (эмпирическая плотность)")
    logger.info("   - Теоретическая плотность F4 с подставленной ОМП θ̂")
    logger.info("   - Вертикальная линия в точке ОМП")
except Exception as e:
    logger.error(f"Ошибка построения графика: {e}")

# --- 12. Манн-Уитни ---
mid = n // 2
group1 = data[:mid]
group2 = data[mid:]
n1_mw, n2_mw = len(group1), len(group2)

u_stat = 0
for x in group1:
    for y in group2:
        if x < y:
            u_stat += 1
        elif x == y:
            u_stat += 0.5

e_u = (n1_mw * n2_mw) / 2
s_u = math.sqrt((n1_mw * n2_mw * (n1_mw + n2_mw + 1)) / 12)
z_mw = (u_stat - e_u) / s_u

logger.info("\n[ЗАДАНИЕ 12] ОДНОРОДНОСТЬ (Критерий Манна-Уитни)")
logger.info(f"1. Выборка разделена на две части: n1 = {n1_mw}, n2 = {n2_mw}")
logger.info(f"2. Статистика U (число инверсий) = {u_stat}")
logger.info(f"3. M[U] = (n1*n2)/2 = {e_u:.2f}")
logger.info(f"4. σ[U] = √[ (n1*n2*(n1+n2+1))/12 ] = {s_u:.4f}")
logger.info(f"5. Z_выч = (U - M[U]) / σ[U] = {z_mw:.4f}")
logger.info(f"6. Критическое Z_крит (α=0.01, двустороннее) = 2.57")
logger.info(
    f"ВЫВОД: {'Однородны' if abs(z_mw) < 2.57 else 'Неоднородны'} (т.к. |Z| {'<' if abs(z_mw) < 2.57 else '>'} 2.57)"
)

logger.info("\n" + "=" * 60)
logger.info("ОТЧЕТ ЗАВЕРШЕН")
logger.info("=" * 60)
