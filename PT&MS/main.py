import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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
sheet_index = 21  # начинается с 0 => для 22 листа берем индекс 21
use_col = 4

try:
    col = pd.read_excel(excel_path, sheet_name=sheet_index, usecols=[use_col])
    series = pd.to_numeric(col.iloc[:, 0], errors="coerce").astype(float)
    data = np.array(series.tolist())
except Exception as e:
    logger.error(f"Ошибка чтения файла: {e}")
    sys.exit()

n = len(data)
logger.info("=" * 70)
logger.info(f"СТАТИСТИЧЕСКИЙ ОТЧЕТ ПО ВЫБОРКЕ (n = {n})")
logger.info("=" * 70)

# =============================================================================
# ЗАДАНИЕ 7. ПЕРВИЧНЫЙ АНАЛИЗ И ВЫДВИЖЕНИЕ ГИПОТЕЗЫ О РАСПРЕДЕЛЕНИИ
# =============================================================================
logger.info("\n" + "=" * 70)
logger.info("[ЗАДАНИЕ 7] ВЫБОРОЧНЫЕ ХАРАКТЕРИСТИКИ И ГИПОТЕЗА О РАСПРЕДЕЛЕНИИ")
logger.info("=" * 70)

# --- 7.1 Расчёт выборочных характеристик ---
logger.info("\n--- 7.1 РАСЧЁТНЫЕ ФОРМУЛЫ И ВЫБОРОЧНЫЕ ХАРАКТЕРИСТИКИ ---")

# Выборочное среднее
# Формула: x̄ = (1/n) * Σxᵢ = np.mean(data)
mean_x = np.mean(data)
logger.info("\n1. ВЫБОРОЧНОЕ СРЕДНЕЕ:")
logger.info("   Формула: x̄ = (1/n) · Σxᵢ")
logger.info(f"   Расчёт:  x̄ = (1/{n}) · {np.sum(data):.4f} = {mean_x:.4f}")

# Выборочная дисперсия
# Формула: S² = (1/(n-1)) * Σ(xᵢ - x̄)² = np.var(data, ddof=1)
var_x = np.var(data, ddof=1)
logger.info("\n2. ВЫБОРОЧНАЯ ДИСПЕРСИЯ:")
logger.info("   Формула: S² = (1/(n-1)) · Σ(xᵢ - x̄)²")
logger.info(f"   Расчёт:  S² = (1/{n - 1}) · {np.sum((data - mean_x) ** 2):.4f} = {var_x:.4f}")

# Выборочное СКО (несмещённое)
# Формула: S = √S² = np.std(data, ddof=1)
std_x = np.std(data, ddof=1)
logger.info("\n3. ВЫБОРОЧНОЕ СКО:")
logger.info("   Формула: S = √S²")
logger.info(f"   Расчёт:  S = √{var_x:.4f} = {std_x:.4f}")

# Медиана
# Формула: x₍med₎ = x₍ₙ/₂₎ при нечётном n, или (x₍ₙ/₂₎ + x₍ₙ/₂+₁₎)/2 при чётном
# Используем: np.median(data)
median_x = np.median(data)
sorted_data = np.sort(data)
logger.info("\n4. МЕДИАНА:")
logger.info("   Формула: x₍med₎ = x₍(n+1)/2₎ (нечётное n) или (x₍n/2₎ + x₍n/2+1₎)/2 (чётное n)")
if n % 2 == 1:
    logger.info(f"   Расчёт:  x₍med₎ = x₍{(n + 1) // 2}₎ = {median_x:.4f}")
else:
    logger.info(f"   Расчёт:  x₍med₎ = (x₍{n // 2}₎ + x₍{n // 2 + 1}₎)/2 = ({sorted_data[n // 2 - 1]:.4f} + {sorted_data[n // 2]:.4f})/2 = {median_x:.4f}")

# Центральные моменты 3-го и 4-го порядка
# µ₃ = (1/n) * Σ(xᵢ - x̄)³ = np.mean((data - mean_x)**3)
# µ₄ = (1/n) * Σ(xᵢ - x̄)⁴ = np.mean((data - mean_x)**4)
m3 = np.mean((data - mean_x) ** 3)
m4 = np.mean((data - mean_x) ** 4)

# Коэффициент асимметрии
# Формула: k₍as₎ = µ₃ / S³ = np.mean((data - mean_x)**3) / np.std(data, ddof=0)**3
# Примечание: для коэффициентов используем смещённое СКО (ddof=0)
std_biased = np.std(data, ddof=0)
As = m3 / (std_biased**3) if std_biased != 0 else 0
logger.info("\n5. КОЭФФИЦИЕНТ АСИММЕТРИИ:")
logger.info("   Формула: k₍as₎ = µ₃ / σ³, где µ₃ = (1/n)·Σ(xᵢ - x̄)³, σ = √[(1/n)·Σ(xᵢ - x̄)²]")
logger.info(f"   Расчёт:  µ₃ = {m3:.6f}")
logger.info(f"            σ = {std_biased:.4f}")
logger.info(f"            k₍as₎ = {m3:.6f} / {std_biased**3:.6f} = {As:.4f}")

# Коэффициент эксцесса
# Формула: k₍ex₎ = µ₄ / S⁴ - 3 = np.mean((data - mean_x)**4) / np.std(data, ddof=0)**4 - 3
Ex = (m4 / (std_biased**4) - 3) if std_biased != 0 else 0
logger.info("\n6. КОЭФФИЦИЕНТ ЭКСЦЕССА:")
logger.info("   Формула: k₍as₎ = µ₄ / σ⁴ - 3, где µ₄ = (1/n)·Σ(xᵢ - x̄)⁴")
logger.info(f"   Расчёт:  µ₄ = {m4:.6f}")
logger.info(f"            k₍as₎ = {m4:.6f} / {std_biased**4:.6f} - 3 = {Ex:.4f}")

# Дополнительные характеристики выборки
x_min = np.min(data)
x_max = np.max(data)
logger.info("\n7. ДОПОЛНИТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ:")
logger.info(f"   Минимум: x_min = {x_min:.4f}")
logger.info(f"   Максимум: x_max = {x_max:.4f}")
logger.info(f"   Размах: R = x_max - x_min = {x_max - x_min:.4f}")

# --- Сводная таблица выборочных характеристик ---
logger.info("\n--- СВОДНАЯ ТАБЛИЦА ВЫБОРОЧНЫХ ХАРАКТЕРИСТИК ---")
logger.info(f"{'Характеристика':<25} | {'Значение':>12}")
logger.info("-" * 42)
logger.info(f"{'Объём выборки (n)':<25} | {n:>12}")
logger.info(f"{'Среднее (x̄)':<25} | {mean_x:>12.4f}")
logger.info(f"{'Дисперсия (S²)':<25} | {var_x:>12.4f}")
logger.info(f"{'СКО (S)':<25} | {std_x:>12.4f}")
logger.info(f"{'Медиана (x₍med₎)':<25} | {median_x:>12.4f}")
logger.info(f"{'Асимметрия (k₍as₎)':<25} | {As:>12.4f}")
logger.info(f"{'Эксцесс (k₍ex₎)':<25} | {Ex:>12.4f}")
logger.info(f"{'Минимум':<25} | {x_min:>12.4f}")
logger.info(f"{'Максимум':<25} | {x_max:>12.4f}")

# --- 7.2 Построение гистограммы относительных частот ---
logger.info("\n--- 7.2 ГИСТОГРАММА ОТНОСИТЕЛЬНЫХ ЧАСТОТ ---")

# Число интервалов по формуле Стэрджеса: k = [log₂(n)] + 1 = int(np.log2(n)) + 1
k_sturges = int(np.log2(n)) + 1
logger.info(f"Число интервалов (формула Стэрджеса): k = [log₂({n})] + 1 = {k_sturges}")

# Построение гистограммы
fig, axes = plt.subplots(1, 1, figsize=(10, 5))

# Гистограмма относительных частот (density=False, weights для относительных частот)
counts, bins, _ = axes.hist(
    data,
    bins=k_sturges,
    weights=np.ones(n) / n,  # Относительные частоты (сумма = 1)
    edgecolor="black",
    color="steelblue",
    alpha=0.7,
)
axes.set_xlabel("x", fontsize=11)
axes.set_ylabel("Относительная частота (nᵢ/n)", fontsize=11)
axes.set_title("Гистограмма относительных частот", fontsize=12)
axes.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("histogram_task7.png", dpi=150)
plt.close(fig)
logger.info("-> Гистограмма сохранена в 'histogram_task7.png'")

# Таблица интервалов и частот
logger.info(f"\n{'Интервал':<20} | {'nᵢ':<6} | {'Отн.частота':<12}")
logger.info("-" * 60)
bin_width = bins[1] - bins[0]
for i in range(k_sturges):
    ni = int(counts[i] * n + 0.5)  # Восстанавливаем абсолютную частоту
    logger.info(f"[{bins[i]:7.3f}; {bins[i + 1]:7.3f}) | {ni:<6} | {counts[i]:<12.4f}")

# --- 7.3 Теоретические характеристики распределений F1-F4 ---
logger.info("\n--- 7.3 ТЕОРЕТИЧЕСКИЕ ХАРАКТЕРИСТИКИ РАСПРЕДЕЛЕНИЙ ---")

logger.info("\n╔════════════════════════════════════════════════════════════════════╗")
logger.info("║                    ГИПОТЕТИЧЕСКИЕ РАСПРЕДЕЛЕНИЯ                    ║")
logger.info("╠════════════════════════════════════════════════════════════════════╣")

# F1: Нормальное N(a, σ²)
logger.info("║ F1: НОРМАЛЬНОЕ N(a, σ²)                                            ║")
logger.info("║     f(x) = (1/√(2πσ²)) · exp(-(x-a)²/(2σ²)), x ∈ ℝ                 ║")
logger.info("║     Теор. характеристики:                                          ║")
logger.info("║       M[X] = a,  D[X] = σ²,  k₍as₎ = 0,  k₍ex₎ = 0                       ║")
logger.info("╠════════════════════════════════════════════════════════════════════╣")

# F2: Равномерное на [-1, 2θ]
logger.info("║ F2: РАВНОМЕРНОЕ U(-1, 2θ)                                          ║")
logger.info("║     f(x) = 1/(2θ+1), x ∈ [-1, 2θ]                                  ║")
logger.info("║     Теор. характеристики:                                          ║")
logger.info("║       M[X] = (2θ-1)/2 = θ - 0.5                                    ║")
logger.info("║       D[X] = (2θ+1)²/12                                            ║")
logger.info("║       k₍as₎ = 0,  k₍ex₎ = -1.2                                           ║")
logger.info("╠════════════════════════════════════════════════════════════════════╣")

# F3: Гамма (Эрланга 2-го порядка)
logger.info("║ F3: ГАММА-РАСПРЕДЕЛЕНИЕ (Эрланга k=2)                              ║")
logger.info("║     f(x) = θ²·x·e^(-θx), x ≥ 0                                     ║")
logger.info("║     Теор. характеристики:                                          ║")
logger.info("║       M[X] = 2/θ,  D[X] = 2/θ²                                     ║")
logger.info("║       k₍as₎ = √2 ≈ 1.414,  k₍ex₎ = 3                                     ║")
logger.info("╠════════════════════════════════════════════════════════════════════╣")

# F4: Степенное на [0, θ]
logger.info("║ F4: СТЕПЕННОЕ (линейное) на [0, θ]                                 ║")
logger.info("║     f(x) = 2x/θ², x ∈ [0, θ]                                       ║")
logger.info("║     Теор. характеристики:                                          ║")
logger.info("║       M[X] = 2θ/3,  D[X] = θ²/18                                   ║")
logger.info("║       k₍as₎ ≈ -0.566,  k₍ex₎ ≈ -0.6                                      ║")
logger.info("╚════════════════════════════════════════════════════════════════════╝")

# --- 7.4 Оценка параметров для каждого распределения ---
logger.info("\n--- 7.4 ОЦЕНКА ПАРАМЕТРОВ И СРАВНЕНИЕ ХАРАКТЕРИСТИК ---")

# Для F1: a = x̄, σ² = S²
a_f1 = mean_x
sigma2_f1 = var_x
As_f1_theor = 0
Ex_f1_theor = 0

# Для F2: из M[X] = θ - 0.5 => θ = x̄ + 0.5
theta_f2 = mean_x + 0.5
var_f2_theor = (2 * theta_f2 + 1) ** 2 / 12
As_f2_theor = 0
Ex_f2_theor = -1.2

# Для F3: из M[X] = 2/θ => θ = 2/x̄
theta_f3 = 2 / mean_x if mean_x > 0 else np.inf
var_f3_theor = 2 / (theta_f3**2) if theta_f3 != np.inf else np.inf
As_f3_theor = np.sqrt(2)
Ex_f3_theor = 3

# Для F4: из M[X] = 2θ/3 => θ = 3x̄/2
theta_f4 = 3 * mean_x / 2
var_f4_theor = theta_f4**2 / 18
As_f4_theor = -0.5657  # Точное значение: -2√2/5 * √(5/2) ≈ -0.5657
Ex_f4_theor = -0.6

logger.info("\n┌──────────────────────────────────────────────────────────────────────┐")
logger.info("│           СРАВНЕНИЕ ВЫБОРОЧНЫХ И ТЕОРЕТИЧЕСКИХ ХАРАКТЕРИСТИК        │")
logger.info("├──────────┬───────────┬───────────┬───────────┬───────────┬──────────┤")
logger.info("│ Распред. │  Параметр │   M[X]    │   D[X]    │    k₍as₎     │    k₍ex₎    │")
logger.info("├──────────┼───────────┼───────────┼───────────┼───────────┼──────────┤")
logger.info(f"│ ВЫБОРКА  │     -     │ {mean_x:9.4f} │ {var_x:9.4f} │ {As:9.4f} │ {Ex:8.4f} │")
logger.info("├──────────┼───────────┼───────────┼───────────┼───────────┼──────────┤")
logger.info(f"│ F1 (Норм)│ a={a_f1:.3f}  │ {a_f1:9.4f} │ {sigma2_f1:9.4f} │ {As_f1_theor:9.4f} │ {Ex_f1_theor:8.4f} │")
logger.info(f"│ F2 (Равн)│ θ={theta_f2:.3f}  │ {theta_f2 - 0.5:9.4f} │ {var_f2_theor:9.4f} │ {As_f2_theor:9.4f} │ {Ex_f2_theor:8.4f} │")
logger.info(f"│ F3 (Гамм)│ θ={theta_f3:.3f}  │ {2 / theta_f3:9.4f} │ {var_f3_theor:9.4f} │ {As_f3_theor:9.4f} │ {Ex_f3_theor:8.4f} │")
logger.info(f"│ F4 (Степ)│ θ={theta_f4:.3f}  │ {2 * theta_f4 / 3:9.4f} │ {var_f4_theor:9.4f} │ {As_f4_theor:9.4f} │ {Ex_f4_theor:8.4f} │")
logger.info("└──────────┴───────────┴───────────┴───────────┴───────────┴──────────┘")

# --- 7.5 Вычисление отклонений и выбор гипотезы ---
logger.info("\n--- 7.5 АНАЛИЗ ОТКЛОНЕНИЙ И ВЫБОР ГИПОТЕЗЫ ---")

# Вычисляем суммарное относительное отклонение для каждого распределения
def calc_deviation(var_theor, As_theor, Ex_theor):
    """Вычисляет суммарное отклонение характеристик"""
    dev_var = abs(var_x - var_theor) / var_x if var_x != 0 else abs(var_x - var_theor)
    dev_As = abs(As - As_theor)
    dev_Ex = abs(Ex - Ex_theor)
    return dev_var + dev_As + dev_Ex, dev_var, dev_As, dev_Ex


dev_f1, dev_var_f1, dev_As_f1, dev_Ex_f1 = calc_deviation(sigma2_f1, As_f1_theor, Ex_f1_theor)
dev_f2, dev_var_f2, dev_As_f2, dev_Ex_f2 = calc_deviation(var_f2_theor, As_f2_theor, Ex_f2_theor)
dev_f3, dev_var_f3, dev_As_f3, dev_Ex_f3 = calc_deviation(var_f3_theor, As_f3_theor, Ex_f3_theor)
dev_f4, dev_var_f4, dev_As_f4, dev_Ex_f4 = calc_deviation(var_f4_theor, As_f4_theor, Ex_f4_theor)

logger.info("\nОтклонения от теоретических значений:")
logger.info(f"{'Распред.':<10} | {'ΔD[X]/D':<10} | {'Δk₍as₎':<10} | {'Δk₍ex₎':<10} | {'Σ откл.':<10}")
logger.info("-" * 60)
logger.info(f"{'F1 (Норм)':<10} | {dev_var_f1:<10.4f} | {dev_As_f1:<10.4f} | {dev_Ex_f1:<10.4f} | {dev_f1:<10.4f}")
logger.info(f"{'F2 (Равн)':<10} | {dev_var_f2:<10.4f} | {dev_As_f2:<10.4f} | {dev_Ex_f2:<10.4f} | {dev_f2:<10.4f}")
logger.info(f"{'F3 (Гамм)':<10} | {dev_var_f3:<10.4f} | {dev_As_f3:<10.4f} | {dev_Ex_f3:<10.4f} | {dev_f3:<10.4f}")
logger.info(f"{'F4 (Степ)':<10} | {dev_var_f4:<10.4f} | {dev_As_f4:<10.4f} | {dev_Ex_f4:<10.4f} | {dev_f4:<10.4f}")

# Определяем лучшее распределение
deviations = {"F1": dev_f1, "F2": dev_f2, "F3": dev_f3, "F4": dev_f4}
best_fit = min(deviations, key=lambda k: deviations[k])

# Дополнительные критерии выбора
logger.info("\n--- 7.6 ВИЗУАЛЬНЫЙ И КАЧЕСТВЕННЫЙ АНАЛИЗ ---")

# Проверка области определения
logger.info("\nПроверка области определения данных:")
logger.info(f"  - Данные: x ∈ [{x_min:.4f}, {x_max:.4f}]")

support_check = {
    "F1": True,  # x ∈ ℝ - всегда подходит
    "F2": x_min >= -1,  # x ∈ [-1, 2θ]
    "F3": x_min >= 0,  # x ≥ 0
    "F4": x_min >= 0,  # x ∈ [0, θ]
}

logger.info(f"  - F1 (ℝ): {'✓ подходит' if support_check['F1'] else '✗ не подходит'}")
logger.info(f"  - F2 ([-1, 2θ]): {'✓ подходит' if support_check['F2'] else '✗ не подходит (x_min < -1)'}")
logger.info(f"  - F3 ([0, +∞)): {'✓ подходит' if support_check['F3'] else '✗ не подходит (x_min < 0)'}")
logger.info(f"  - F4 ([0, θ]): {'✓ подходит' if support_check['F4'] else '✗ не подходит (x_min < 0)'}")

# Анализ асимметрии
logger.info(f"\nАнализ асимметрии (k₍as₎ = {As:.4f}):")
if abs(As) < 0.25:
    logger.info("  -> Распределение близко к симметричному (подходит F1, F2)")
elif As > 0.5:
    logger.info("  -> Положительная асимметрия (правый хвост длиннее) - подходит F3")
elif As < -0.25:
    logger.info("  -> Отрицательная асимметрия (левый хвост длиннее) - подходит F4")

# Анализ эксцесса
logger.info(f"\nАнализ эксцесса (k₍ex₎ = {Ex:.4f}):")
if abs(Ex) < 0.5:
    logger.info("  -> Эксцесс близок к нормальному (подходит F1)")
elif Ex < -0.5:
    logger.info("  -> Отрицательный эксцесс (плосковершинное) - подходит F2 или F4")
elif Ex > 1:
    logger.info("  -> Положительный эксцесс (островершинное) - подходит F3")

# --- ИТОГОВЫЙ ВЫВОД ---
logger.info("\n" + "=" * 70)
logger.info("                          ИТОГОВЫЙ ВЫВОД")
logger.info("=" * 70)

# Финальный выбор с учётом всех критериев
final_choice = best_fit
for dist in ["F1", "F2", "F3", "F4"]:
    if not support_check.get(dist, True) and dist == best_fit:
        # Если лучшее по отклонениям не подходит по области определения
        remaining = {k: v for k, v in deviations.items() if support_check.get(k, True)}
        if remaining:
            final_choice = min(remaining, key=lambda k: remaining[k])

dist_names = {
    "F1": "Нормальное N(a, σ²)",
    "F2": "Равномерное U(-1, 2θ)",
    "F3": "Гамма (Эрланга k=2)",
    "F4": "Степенное (линейное)",
}

logger.info("\nНа основании:")
logger.info("  1. Визуального анализа гистограммы")
logger.info("  2. Сравнения выборочных характеристик с теоретическими")
logger.info("  3. Анализа области определения данных")
logger.info("  4. Анализа коэффициентов асимметрии и эксцесса")
logger.info("\n╔══════════════════════════════════════════════════════════════════╗")
logger.info(f"║  ГИПОТЕЗА: Данные подчиняются распределению {final_choice}               ║")
logger.info(f"║  {dist_names[final_choice]:<62} ║")
logger.info("╚══════════════════════════════════════════════════════════════════╝")

# Свойства выбранного распределения
logger.info(f"\nСвойства гипотетического распределения {final_choice}:")
if final_choice == "F1":
    logger.info(f"  • Симметричное относительно математического ожидания a = {a_f1:.4f}")
    logger.info("  • Колоколообразная форма плотности")
    logger.info(f"  • 68% данных в интервале [a-σ, a+σ] = [{a_f1 - np.sqrt(sigma2_f1):.4f}, {a_f1 + np.sqrt(sigma2_f1):.4f}]")
    logger.info("  • Асимметрия = 0, эксцесс = 0")
elif final_choice == "F2":
    logger.info(f"  • Равномерное распределение на интервале [-1, {2 * theta_f2:.4f}]")
    logger.info(f"  • Плотность постоянна: f(x) = 1/{2 * theta_f2 + 1:.4f}")
    logger.info("  • Симметричное (k₍as₎ = 0), плосковершинное (k₍ex₎ = -1.2)")
elif final_choice == "F3":
    logger.info(f"  • Гамма-распределение (Эрланга 2-го порядка) с параметром θ = {theta_f3:.4f}")
    logger.info("  • Область определения: x ≥ 0")
    logger.info("  • Положительная асимметрия (k₍as₎ ≈ 1.414) - правый хвост длиннее")
    logger.info("  • Положительный эксцесс (k₍ex₎ = 3) - островершинное")
elif final_choice == "F4":
    logger.info(f"  • Степенное распределение с параметром θ = {theta_f4:.4f}")
    logger.info(f"  • Область определения: x ∈ [0, {theta_f4:.4f}]")
    logger.info(f"  • Плотность линейно возрастает от 0 до 2/θ = {2 / theta_f4:.4f}")
    logger.info("  • Отрицательная асимметрия (k₍as₎ ≈ -0.57) - левый хвост длиннее")
    logger.info("  • Отрицательный эксцесс (k₍ex₎ ≈ -0.6) - плосковершинное")

# Сохраняем выбранное распределение для дальнейших заданий
hypothesis = final_choice

# Инициализируем переменные для всех распределений
theta = 0.0
a_mle = mean_x
sigma2_mle = np.var(data, ddof=0)

if hypothesis == "F4":
    theta = theta_f4
elif hypothesis == "F3":
    theta = theta_f3
elif hypothesis == "F2":
    theta = theta_f2

logger.info("\n" + "=" * 70)
logger.info("[ЗАДАНИЕ 7] ЗАВЕРШЕНО")
logger.info("=" * 70)

# =============================================================================
# ЗАДАНИЕ 8. КРИТЕРИЙ СЕРИЙ (ПРОВЕРКА СЛУЧАЙНОСТИ)
# =============================================================================
logger.info("\n" + "=" * 70)
logger.info("[ЗАДАНИЕ 8] ПРОВЕРКА СЛУЧАЙНОСТИ (Критерий серий)")
logger.info("=" * 70)

signs = np.where(data > median_x, "+", np.where(data < median_x, "-", ""))
signs = signs[signs != ""]

n1 = np.sum(signs == "+")
n2 = np.sum(signs == "-")
ks = 1 + np.sum(signs[1:] != signs[:-1])

numerator = (ks - ((2 * n1 * n2) / (n1 + n2)) - 1) - 0.5
denominator = (2 * n1 * n2 * (2 * n1 * n2 - (n1 + n2))) / ((n1 + n2)**2 * (n1 + n2 - 1))
sqrt_denom = np.sqrt(denominator)

z_calc_series = numerator / sqrt_denom

logger.info(f"\nМедиана: {median_x}")
logger.info(f"1. Число элементов > медианы (n1) = {n1}")
logger.info(f"2. Число элементов < медианы (n2) = {n2}")
logger.info(f"3. Количество серий (KS) = {ks}")
logger.info(f"4. Числитель = (KS - (2·n1·n2)/(n1+n2) - 1) - 0.5 = {numerator:.4f}")
logger.info(f"5. Знаменатель = √[(2n1n2(2n1n2-(n1+n2)))/((n1+n2)²·(n1+n2-1))] = {sqrt_denom:.4f}")
logger.info(f"6. Z_выч = {numerator:.4f} / {sqrt_denom:.4f} = {z_calc_series:.4f}")
logger.info("7. Критическое Z_крит (α=0.05) = 1.96")
logger.info(f"\nВЫВОД: Выборка {'СЛУЧАЙНА' if abs(z_calc_series) < 1.96 else 'НЕ СЛУЧАЙНА'} (т.к. |Z| = {abs(z_calc_series):.4f} {'<' if abs(z_calc_series) < 1.96 else '>'} 1.96)")

# =============================================================================
# ЗАДАНИЕ 9. ОЦЕНКА МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ (ОМП)
# =============================================================================
logger.info("\n" + "=" * 70)
logger.info(f"[ЗАДАНИЕ 9] ОЦЕНКА МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ (ОМП) для {hypothesis}")
logger.info("=" * 70)

if hypothesis == "F4":
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
    logger.info("   - L(θ) = 0 при θ < max(xᵢ)")
    logger.info("   - L(θ) убывает при θ ≥ max(xᵢ)")
    logger.info("   - Максимум достигается при минимальном допустимом θ")
    logger.info("\n5. ВЫВОД: ОМП для θ:")
    logger.info("   θ̂ = max(x₁, x₂, ..., xₙ) = X₍ₙ₎")
    theta_mle = np.max(data)
    logger.info("\n--- Численный расчет ---")
    logger.info(f"   θ̂ = max(X) = {theta_mle:.4f}")
    log_likelihood = n * np.log(2) + np.sum(np.log(data)) - 2 * n * np.log(theta_mle)
    logger.info(f"   ln L(θ̂) = {log_likelihood:.4f}")
    theta = theta_mle
elif hypothesis == "F3":
    logger.info("\nГипотетическое распределение F3:")
    logger.info("  Плотность: f(x; θ) = θ²·x·e^(-θx),  x ≥ 0")
    logger.info("  Неизвестный параметр: θ > 0")
    logger.info("\n--- Вывод ОМП для параметра θ ---")
    logger.info("\n1. Функция правдоподобия:")
    logger.info("   L(θ) = ∏ᵢ θ²·xᵢ·e^(-θxᵢ) = θ^(2n) · (∏ᵢxᵢ) · e^(-θ·Σxᵢ)")
    logger.info("\n2. Логарифм функции правдоподобия:")
    logger.info("   ln L(θ) = 2n·ln(θ) + Σᵢln(xᵢ) - θ·Σxᵢ")
    logger.info("\n3. Уравнение правдоподобия:")
    logger.info("   d[ln L(θ)]/dθ = 2n/θ - Σxᵢ = 0")
    logger.info("   => θ̂ = 2n / Σxᵢ = 2 / x̄")
    theta_mle = 2 / mean_x
    logger.info("\n--- Численный расчет ---")
    logger.info(f"   θ̂ = 2 / x̄ = 2 / {mean_x:.4f} = {theta_mle:.4f}")
    theta = theta_mle
elif hypothesis == "F2":
    logger.info("\nГипотетическое распределение F2:")
    logger.info("  Плотность: f(x; θ) = 1/(2θ+1),  x ∈ [-1, 2θ]")
    logger.info("  Неизвестный параметр: θ > 0")
    logger.info("\n--- Вывод ОМП для параметра θ ---")
    logger.info("\n1. Функция правдоподобия:")
    logger.info("   L(θ) = (1/(2θ+1))ⁿ при -1 ≤ xᵢ ≤ 2θ для всех i")
    logger.info("\n2. Анализ:")
    logger.info("   L(θ) убывает по θ, поэтому θ должно быть минимальным")
    logger.info("   Ограничение: 2θ ≥ max(xᵢ) => θ ≥ max(xᵢ)/2")
    logger.info("\n3. ВЫВОД: ОМП для θ:")
    logger.info("   θ̂ = max(xᵢ) / 2")
    theta_mle = np.max(data) / 2
    logger.info("\n--- Численный расчет ---")
    logger.info(f"   θ̂ = max(X) / 2 = {np.max(data):.4f} / 2 = {theta_mle:.4f}")
    theta = theta_mle
else:
    logger.info("\nГипотетическое распределение F1 (Нормальное):")
    logger.info("  Плотность: f(x; a,σ) = (1/√(2πσ²))·exp(-(x-a)²/(2σ²))")
    logger.info("  Неизвестные параметры: a ∈ ℝ, σ > 0")
    logger.info("\n--- Вывод ОМП ---")
    logger.info("\n1. Уравнения правдоподобия дают:")
    logger.info("   â = x̄ (выборочное среднее)")
    logger.info("   σ̂² = (1/n)·Σ(xᵢ - x̄)² (смещённая дисперсия)")
    logger.info("\n--- Численный расчет ---")
    logger.info(f"   â = {a_mle:.4f}")
    logger.info(f"   σ̂² = {sigma2_mle:.4f}")
    logger.info(f"   σ̂ = {np.sqrt(sigma2_mle):.4f}")

# =============================================================================
# ЗАДАНИЕ 10. КРИТЕРИЙ ХИ-КВАДРАТ
# =============================================================================
logger.info("\n" + "=" * 70)
logger.info(f"[ЗАДАНИЕ 10] КРИТЕРИЙ ХИ-КВАДРАТ (α=0.1, Распределение {hypothesis})")
logger.info("=" * 70)

k_sturges = int(np.log2(n)) + 1
chi_sq = 0

if hypothesis == "F4":
    step = theta / k_sturges
    logger.info(f"\nИспользуется ОМП: θ̂ = {theta:.4f}")
    logger.info("Формула плотности: f(x) = 2x / θ̂²")
    logger.info(f"Число интервалов (по Стёрджесу): k = {k_sturges}")
    logger.info(f"\n{'Интервал':<20} | {'nᵢ':<6} | {'pᵢ':<8} | {'nᵢ теор':<8} | {'(nᵢ-nᵢт)²/nᵢт':<12}")
    logger.info("-" * 70)
    for i in range(k_sturges):
        left = i * step
        right = (i + 1) * step
        ni_obs = np.sum((data >= left) & (data < right)) if i < k_sturges - 1 else np.sum((data >= left) & (data <= right))
        p_i = (right**2 - left**2) / (theta**2)
        ni_theor = n * p_i
        term = ((ni_obs - ni_theor) ** 2) / ni_theor if ni_theor > 0 else 0
        chi_sq += term
        logger.info(f"[{left:6.3f}; {right:6.3f}) | {ni_obs:6d} | {p_i:8.4f} | {ni_theor:8.2f} | {term:12.4f}")
elif hypothesis == "F3":
    bins_edges = np.linspace(0, np.max(data) * 1.1, k_sturges + 1)
    logger.info(f"\nИспользуется ОМП: θ̂ = {theta:.4f}")
    logger.info("Формула плотности: f(x) = θ̂²·x·e^(-θ̂x)")
    logger.info(f"Число интервалов: k = {k_sturges}")
    logger.info(f"\n{'Интервал':<20} | {'nᵢ':<6} | {'pᵢ':<8} | {'nᵢ теор':<8} | {'(nᵢ-nᵢт)²/nᵢт':<12}")
    logger.info("-" * 70)
    for i in range(k_sturges):
        left, right = bins_edges[i], bins_edges[i + 1]
        ni_obs = np.sum((data >= left) & (data < right)) if i < k_sturges - 1 else np.sum((data >= left) & (data <= right))
        F_left = 1 - (1 + theta * left) * np.exp(-theta * left)
        F_right = 1 - (1 + theta * right) * np.exp(-theta * right)
        p_i = F_right - F_left
        ni_theor = n * p_i
        term = ((ni_obs - ni_theor) ** 2) / ni_theor if ni_theor > 0 else 0
        chi_sq += term
        logger.info(f"[{left:6.3f}; {right:6.3f}) | {ni_obs:6d} | {p_i:8.4f} | {ni_theor:8.2f} | {term:12.4f}")
elif hypothesis == "F2":
    bins_edges = np.linspace(-1, 2 * theta, k_sturges + 1)
    logger.info(f"\nИспользуется ОМП: θ̂ = {theta:.4f}")
    logger.info(f"Формула плотности: f(x) = 1/(2θ̂+1) = {1 / (2 * theta + 1):.4f}")
    logger.info(f"Число интервалов: k = {k_sturges}")
    logger.info(f"\n{'Интервал':<20} | {'nᵢ':<6} | {'pᵢ':<8} | {'nᵢ теор':<8} | {'(nᵢ-nᵢт)²/nᵢт':<12}")
    logger.info("-" * 70)
    for i in range(k_sturges):
        left, right = bins_edges[i], bins_edges[i + 1]
        ni_obs = np.sum((data >= left) & (data < right)) if i < k_sturges - 1 else np.sum((data >= left) & (data <= right))
        p_i = (right - left) / (2 * theta + 1)
        ni_theor = n * p_i
        term = ((ni_obs - ni_theor) ** 2) / ni_theor if ni_theor > 0 else 0
        chi_sq += term
        logger.info(f"[{left:6.3f}; {right:6.3f}) | {ni_obs:6d} | {p_i:8.4f} | {ni_theor:8.2f} | {term:12.4f}")
else:
    bins_edges = np.linspace(np.min(data), np.max(data), k_sturges + 1)
    logger.info(f"\nИспользуется ОМП: â = {a_mle:.4f}, σ̂ = {np.sqrt(sigma2_mle):.4f}")
    logger.info(f"Число интервалов: k = {k_sturges}")
    logger.info(f"\n{'Интервал':<20} | {'nᵢ':<6} | {'pᵢ':<8} | {'nᵢ теор':<8} | {'(nᵢ-nᵢт)²/nᵢт':<12}")
    logger.info("-" * 70)
    for i in range(k_sturges):
        left, right = bins_edges[i], bins_edges[i + 1]
        ni_obs = np.sum((data >= left) & (data < right)) if i < k_sturges - 1 else np.sum((data >= left) & (data <= right))
        p_i = stats.norm.cdf(right, a_mle, np.sqrt(sigma2_mle)) - stats.norm.cdf(left, a_mle, np.sqrt(sigma2_mle))
        ni_theor = n * p_i
        term = ((ni_obs - ni_theor) ** 2) / ni_theor if ni_theor > 0 else 0
        chi_sq += term
        logger.info(f"[{left:6.3f}; {right:6.3f}) | {ni_obs:6d} | {p_i:8.4f} | {ni_theor:8.2f} | {term:12.4f}")

r = 1 if hypothesis in ["F2", "F3", "F4"] else 2
nu = k_sturges - r - 1
logger.info("-" * 70)
logger.info(f"\n1. Хи-квадрат выч = Σ[(nᵢ-nᵢт)²/nᵢт] = {chi_sq:.4f}")
logger.info(f"2. Число оцениваемых параметров r = {r}")
logger.info(f"3. Степени свободы ν = k - r - 1 = {k_sturges} - {r} - 1 = {nu}")
logger.info(f"4. Для α=0.1 и ν={nu} найди χ²_крит в таблице")

# =============================================================================
# ЗАДАНИЕ 11. ГРАФИК ГИСТОГРАММЫ И ПЛОТНОСТИ
# =============================================================================
logger.info("\n" + "=" * 70)
logger.info("[ЗАДАНИЕ 11] ГРАФИК ПЛОТНОСТИ И ГИСТОГРАММЫ")
logger.info("=" * 70)

try:
    fig, ax = plt.subplots(figsize=(12, 7))
    counts, bins, patches = ax.hist(data, bins=k_sturges, density=True, edgecolor="black", color="skyblue", alpha=0.7, label="Гистограмма (эмпирическая плотность)")
    if hypothesis == "F4":
        x_vals = np.linspace(0, theta, 500)
        f_x = (2 * x_vals) / (theta**2)
        label_str = rf"Плотность $F_4$: $f(x) = \frac{{2x}}{{\hat{{\theta}}^2}}$, $\hat{{\theta}}={theta:.4f}$"
    elif hypothesis == "F3":
        x_vals = np.linspace(0, np.max(data) * 1.2, 500)
        f_x = (theta**2) * x_vals * np.exp(-theta * x_vals)
        label_str = rf"Плотность $F_3$: $f(x) = \hat{{\theta}}^2 x e^{{-\hat{{\theta}}x}}$, $\hat{{\theta}}={theta:.4f}$"
    elif hypothesis == "F2":
        x_vals = np.linspace(-1, 2 * theta, 500)
        f_x = np.ones_like(x_vals) / (2 * theta + 1)
        label_str = rf"Плотность $F_2$: $f(x) = \frac{{1}}{{2\hat{{\theta}}+1}}$, $\hat{{\theta}}={theta:.4f}$"
    else:
        x_vals = np.linspace(np.min(data) - std_x, np.max(data) + std_x, 500)
        f_x = stats.norm.pdf(x_vals, a_mle, np.sqrt(sigma2_mle))
        label_str = rf"Плотность $F_1$: $N(\hat{{a}}, \hat{{\sigma}}^2)$, $\hat{{a}}={a_mle:.4f}$, $\hat{{\sigma}}={np.sqrt(sigma2_mle):.4f}$"
    ax.plot(x_vals, f_x, "r-", lw=2.5, label=label_str)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Плотность вероятности f(x)", fontsize=12)
    ax.set_title(f"Гистограмма выборки и теоретическая плотность {hypothesis}", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(np.max(counts) * 1.1, np.max(f_x) * 1.1))
    textstr = f"n = {n}\nГипотеза: {hypothesis}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)
    plt.tight_layout()
    fig.savefig("histogram.png", dpi=150)
    plt.close(fig)
    logger.info("\n-> График успешно сохранен в 'histogram.png'")
except Exception as e:
    logger.error(f"Ошибка построения графика: {e}")

# =============================================================================
# ЗАДАНИЕ 12. КРИТЕРИЙ МАННА-УИТНИ (ОДНОРОДНОСТЬ)
# =============================================================================
logger.info("\n" + "=" * 70)
logger.info("[ЗАДАНИЕ 12] ОДНОРОДНОСТЬ (Критерий Манна-Уитни)")
logger.info("=" * 70)

mid = n // 2
group1 = data[:mid]
group2 = data[mid:]
n_mw, m_mw = len(group1), len(group2)

comparison = group1[:, np.newaxis] < group2
ties = group1[:, np.newaxis] == group2
u_stat = np.sum(comparison) + 0.5 * np.sum(ties)

numerator_mw = u_stat - (n_mw * m_mw) / 2
denominator_mw = np.sqrt((n_mw * m_mw * (n_mw + m_mw + 1)) / 12)
z_mw = numerator_mw / denominator_mw

logger.info(f"\n1. Выборка разделена на две части: n1 = {n_mw}, n2 = {m_mw}")
logger.info(f"2. Статистика U (число инверсий) = {u_stat}")
logger.info(f"3. Числитель = U - (n·m)/2 = {numerator_mw:.2f}")
logger.info(f"4. Знаменатель = √[(n·m·(n+m+1))/12] = {denominator_mw:.4f}")
logger.info(f"5. Z_выч = (U - (n·m)/2) / √[(n·m·(n+m+1))/12] = {z_mw:.4f}")
logger.info("6. Критическое Z_крит (α=0.01, двустороннее) = 2.57")
logger.info(f"\nВЫВОД: Выборки {'ОДНОРОДНЫ' if abs(z_mw) < 2.57 else 'НЕОДНОРОДНЫ'} (т.к. |Z| = {abs(z_mw):.4f} {'<' if abs(z_mw) < 2.57 else '>'} 2.57)")

# =============================================================================
# ОТЧЕТ ЗАВЕРШЕН
# =============================================================================
logger.info("\n" + "=" * 70)
logger.info("ОТЧЕТ ЗАВЕРШЕН")
logger.info("=" * 70)
