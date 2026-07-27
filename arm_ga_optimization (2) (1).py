
import time
import random
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import os

random.seed(42)
np.random.seed(42)

# =====================================================================
# 1. LOAD DATASET
# =====================================================================
# Dataset transaksi ritel asli (Market Basket Optimisation, 7.501 transaksi,
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "Market_Basket_Optimisation_Indonesia.xlsx")
DATASET_SHEET = "Transaksi"
DATASET_KOLOM_ITEM = "Items (Indonesia)"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"File dataset tidak ditemukan di: {DATASET_PATH}\n"
        f"Pastikan '{os.path.basename(DATASET_PATH)}' ada di folder yang sama dengan script ini."
    )

df_raw = pd.read_excel(DATASET_PATH, sheet_name=DATASET_SHEET)

if DATASET_KOLOM_ITEM not in df_raw.columns:
    # Coba cari kolom yang mirip (jaga-jaga kalau file Excel-nya versi beda)
    kandidat = [c for c in df_raw.columns if "indonesia" in str(c).lower()]
    if kandidat:
        print(f"Kolom '{DATASET_KOLOM_ITEM}' tidak ditemukan persis, memakai kolom serupa: '{kandidat[0]}'")
        DATASET_KOLOM_ITEM = kandidat[0]
    else:
        raise KeyError(
            f"Kolom '{DATASET_KOLOM_ITEM}' tidak ditemukan di sheet '{DATASET_SHEET}'.\n"
            f"Kolom yang tersedia di file kamu: {list(df_raw.columns)}\n"
            f"Pastikan file '{os.path.basename(DATASET_PATH)}' yang dipakai adalah versi terbaru "
            f"(hasil terjemahan Market_Basket_Optimisation_Indonesia.xlsx)."
        )

# Kolom item berisi item per transaksi dipisah koma -> ubah jadi list of list
transactions = df_raw[DATASET_KOLOM_ITEM].apply(lambda x: [i.strip() for i in str(x).split(",")]).tolist()

print(f"Jumlah transaksi   : {len(transactions)}")
print(f"Contoh transaksi 1 : {transactions[0]}")

# One-hot encoding transaksi (dibutuhkan Apriori & untuk hitung support GA)
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)
items_list = list(te.columns_)
n_items = len(items_list)
n_trans = len(transactions)

print(f"Jumlah item unik   : {n_items}")

# Parameter minimum threshold (dipakai konsisten di kedua metode agar adil dibandingkan)
# Nilai lebih kecil dari dataset sintetis sebelumnya, karena dataset ini jauh lebih besar
# (7.501 transaksi, 119 item unik) sehingga kombinasi antar-item lebih "tersebar".
MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.3


# =====================================================================
# 2. BASELINE: ASSOCIATION RULE MINING STANDAR (APRIORI)
# =====================================================================
def run_apriori_baseline():
    start = time.time()
    frequent_itemsets = apriori(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    if len(frequent_itemsets) == 0:
        return pd.DataFrame(), 0.0

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules.sort_values("confidence", ascending=False).reset_index(drop=True)
    runtime = time.time() - start
    return rules, runtime


# =====================================================================
# 3. OPTIMASI: ASSOCIATION RULE MINING + ALGORITMA GENETIKA
# =====================================================================
# Ide dasar: alih-alih menelusuri semua kombinasi itemset secara exhaustive
# seperti Apriori, GA mencari itemset yang "bagus" (support & confidence tinggi)
# lewat proses evolusi (seleksi, crossover, mutasi) -> lebih efisien untuk
# ruang pencarian yang besar.

# --- Parameter GA ---
POP_SIZE = 150
N_GENERATIONS = 100
CROSSOVER_RATE = 0.8
# Mutation rate dibuat menyesuaikan jumlah item, supaya rata-rata cuma ~1-2 bit
# yang berubah tiap mutasi (bukan proporsi tetap yang bisa jadi terlalu ekstrem
# kalau jumlah item banyak, seperti pada dataset 118 item ini).
MUTATION_RATE = round(min(0.12, 1.5 / n_items), 3)
ELITISM = 6
RANDOM_IMMIGRANTS = 15  # individu baru acak tiap generasi, jaga keberagaman populasi
MAX_ITEMSET_SIZE = 5  # batasi ukuran itemset agar tetap interpretable

# Bobot fitness: kombinasi support, confidence, dan lift
W_SUPPORT = 0.4
W_CONFIDENCE = 0.4
W_LIFT = 0.2


def random_chromosome():
    """Buat kromosom biner acak (1 = item ikut dalam itemset)."""
    chromosome = [0] * n_items
    # Pastikan minimal 2 item aktif di awal
    n_active = random.randint(2, min(MAX_ITEMSET_SIZE, n_items))
    active_idx = random.sample(range(n_items), n_active)
    for idx in active_idx:
        chromosome[idx] = 1
    return chromosome


def repair(chromosome):
    """Perbaiki kromosom agar ukuran itemset tetap valid (2..MAX_ITEMSET_SIZE)."""
    active_idx = [i for i, bit in enumerate(chromosome) if bit == 1]
    if len(active_idx) < 2:
        # tambahkan item acak sampai minimal 2
        candidates = [i for i in range(n_items) if i not in active_idx]
        need = 2 - len(active_idx)
        for idx in random.sample(candidates, min(need, len(candidates))):
            chromosome[idx] = 1
    elif len(active_idx) > MAX_ITEMSET_SIZE:
        # buang item acak sampai <= MAX_ITEMSET_SIZE
        excess = len(active_idx) - MAX_ITEMSET_SIZE
        for idx in random.sample(active_idx, excess):
            chromosome[idx] = 0
    return chromosome


def decode(chromosome):
    """Ubah kromosom biner jadi daftar nama item."""
    return [items_list[i] for i, bit in enumerate(chromosome) if bit == 1]


def compute_support(itemset):
    if len(itemset) == 0:
        return 0.0
    mask = df_encoded[itemset].all(axis=1)
    return mask.sum() / n_trans


def best_rule_from_itemset(itemset):
    """
    Dari satu itemset, cari pemisahan antecedent -> consequent (1 item)
    yang menghasilkan confidence tertinggi.
    """
    if len(itemset) < 2:
        return None

    itemset_support = compute_support(itemset)
    if itemset_support == 0:
        return None

    best = None
    for consequent in itemset:
        antecedent = [i for i in itemset if i != consequent]
        antecedent_support = compute_support(antecedent)
        if antecedent_support == 0:
            continue
        confidence = itemset_support / antecedent_support
        consequent_support = compute_support([consequent])
        lift = confidence / consequent_support if consequent_support > 0 else 0

        if best is None or confidence > best["confidence"]:
            best = {
                "antecedents": frozenset(antecedent),
                "consequents": frozenset([consequent]),
                "support": itemset_support,
                "confidence": confidence,
                "lift": lift,
            }
    return best


def fitness(chromosome):
    itemset = decode(chromosome)
    rule = best_rule_from_itemset(itemset)
    if rule is None:
        return 0.0, None
    # Normalisasi lift secara kasar (lift bisa > 1 tanpa batas atas pasti)
    norm_lift = min(rule["lift"] / 5.0, 1.0)
    score = (W_SUPPORT * rule["support"]
             + W_CONFIDENCE * rule["confidence"]
             + W_LIFT * norm_lift)
    return score, rule


def tournament_selection(population, fitnesses, k=3):
    idxs = random.sample(range(len(population)), k)
    best_idx = max(idxs, key=lambda i: fitnesses[i])
    return population[best_idx][:]


def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]
    point = random.randint(1, n_items - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(chromosome):
    for i in range(n_items):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


def run_ga_arm():
    start = time.time()

    population = [repair(random_chromosome()) for _ in range(POP_SIZE)]
    convergence = []  # simpan best fitness tiap generasi
    all_rules_found = {}  # key: (antecedent, consequent) -> rule dict terbaik

    for gen in range(N_GENERATIONS):
        fitness_results = [fitness(ind) for ind in population]
        fitnesses = [f[0] for f in fitness_results]

        # simpan rule valid yang ditemukan sepanjang evolusi
        for score, rule in fitness_results:
            if rule is None:
                continue
            if rule["support"] >= MIN_SUPPORT and rule["confidence"] >= MIN_CONFIDENCE:
                key = (rule["antecedents"], rule["consequents"])
                if key not in all_rules_found or score > all_rules_found[key][0]:
                    all_rules_found[key] = (score, rule)

        convergence.append(max(fitnesses))

        # Elitism: individu terbaik langsung lolos ke generasi berikutnya
        elite_idx = np.argsort(fitnesses)[-ELITISM:]
        new_population = [population[i][:] for i in elite_idx]

        # Random immigrants: masukkan individu baru acak agar populasi tetap beragam
        for _ in range(RANDOM_IMMIGRANTS):
            if len(new_population) < POP_SIZE:
                new_population.append(repair(random_chromosome()))

        # Isi sisa populasi lewat seleksi + crossover + mutasi
        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = repair(mutate(child1))
            child2 = repair(mutate(child2))
            new_population.append(child1)
            if len(new_population) < POP_SIZE:
                new_population.append(child2)

        population = new_population

    runtime = time.time() - start

    # Susun hasil akhir jadi DataFrame, mirip format output association_rules()
    rows = []
    for score, rule in all_rules_found.values():
        rows.append({
            "antecedents": rule["antecedents"],
            "consequents": rule["consequents"],
            "support": rule["support"],
            "confidence": rule["confidence"],
            "lift": rule["lift"],
            "fitness": score,
        })
    rules_df = pd.DataFrame(rows)
    if len(rules_df) > 0:
        rules_df = rules_df.sort_values("confidence", ascending=False).reset_index(drop=True)

    return rules_df, runtime, convergence


# =====================================================================
# 4. INPUT MANUAL DARI USER
# =====================================================================

def input_angka(prompt, default, tipe=float):
    """Ambil input angka dari user, pakai nilai default kalau dikosongkan."""
    val = input(f"{prompt} [default={default}]: ").strip()
    if val == "":
        return default
    try:
        return tipe(val)
    except ValueError:
        print(f"  -> input tidak valid, pakai default {default}")
        return default


def pilih_metode():
    print("\n=== PILIH METODE YANG INGIN DIJALANKAN ===")
    print("1. ARM biasa saja (baseline / sebelum optimasi)")
    print("2. ARM + Algoritma Genetika saja (setelah optimasi)")
    print("3. Jalankan keduanya lalu bandingkan (disarankan)")
    pilihan = input("Pilih (1/2/3) [default=3]: ").strip()
    if pilihan not in ("1", "2", "3"):
        pilihan = "3"
    return pilihan


def atur_threshold_arm():
    """Threshold ARM dipakai baik oleh Apriori maupun GA, biar perbandingan adil."""
    print("\n--- Threshold Association Rule Mining ---")
    min_sup = input_angka("Minimum support (0-1)", MIN_SUPPORT, float)
    min_conf = input_angka("Minimum confidence (0-1)", MIN_CONFIDENCE, float)
    return min_sup, min_conf


def atur_parameter_ga():
    """Parameter Algoritma Genetika diatur manual sebelum proses optimasi jalan."""
    print("\n--- Parameter Algoritma Genetika (optimasi) ---")
    pop = input_angka("Ukuran populasi", POP_SIZE, int)
    gen = input_angka("Jumlah generasi", N_GENERATIONS, int)
    cr = input_angka("Crossover rate (0-1)", CROSSOVER_RATE, float)
    mr = input_angka("Mutation rate (0-1)", MUTATION_RATE, float)
    return pop, gen, cr, mr


def tampilkan_rekomendasi(baseline_rules, ga_rules, metode_dipilih):
    """Ubah rule dengan confidence/lift tertinggi jadi rekomendasi bundling/promo yang mudah dibaca."""
    # Prioritas: pakai rule dari GA kalau ada, kalau tidak pakai baseline
    if metode_dipilih in ("2", "3") and len(ga_rules) > 0:
        sumber, rules = "ARM + GA (Optimasi)", ga_rules
    elif len(baseline_rules) > 0:
        sumber, rules = "Apriori (Baseline)", baseline_rules
    else:
        print("\nBelum ada rule yang bisa dijadikan rekomendasi.")
        return

    top = rules.sort_values("confidence", ascending=False).head(5)
    print(f"\n=== REKOMENDASI PRODUK (berdasarkan {sumber}) ===")
    for i, row in enumerate(top.itertuples(), start=1):
        antecedent = ", ".join(sorted(row.antecedents))
        consequent = ", ".join(sorted(row.consequents))
        print(f"{i}. Pelanggan yang beli [{antecedent}] cenderung juga beli [{consequent}]")
        print(f"   -> confidence={row.confidence:.2%}, lift={row.lift:.2f}, support={row.support:.2%}")
    print("\nSaran: item-item ini bisa ditempatkan berdekatan di rak atau dijadikan paket promo.")


# =====================================================================
# 5. FUNGSI RINGKAS & SIMPAN HASIL
# =====================================================================
def summarize(rules_df, runtime, method_name):
    if len(rules_df) == 0:
        return {
            "Metode": method_name, "Waktu (detik)": round(runtime, 4),
            "Jumlah Rule": 0, "Rata-rata Support": None,
            "Rata-rata Confidence": None, "Rata-rata Lift": None,
            "Confidence Maksimum": None,
        }
    return {
        "Metode": method_name,
        "Waktu (detik)": round(runtime, 4),
        "Jumlah Rule": len(rules_df),
        "Rata-rata Support": round(rules_df["support"].mean(), 4),
        "Rata-rata Confidence": round(rules_df["confidence"].mean(), 4),
        "Rata-rata Lift": round(rules_df["lift"].mean(), 4),
        "Confidence Maksimum": round(rules_df["confidence"].max(), 4),
    }


def rules_to_readable(rules_df):
    if len(rules_df) == 0:
        return pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift"])
    out = rules_df.copy()
    out["Antecedent"] = out["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    out["Consequent"] = out["consequents"].apply(lambda s: ", ".join(sorted(s)))
    cols = ["Antecedent", "Consequent", "support", "confidence", "lift"]
    out = out[cols].rename(columns={"support": "Support", "confidence": "Confidence", "lift": "Lift"})
    return out.round(4)


def tampilkan_daftar_rule(baseline_rules, ga_rules, metode_dipilih):
    """Cetak rule dalam bentuk nama barang yang saling terhubung, bukan cuma angka statistik."""

    def cetak_rules(judul, rules_df):
        print(f"\n--- {judul} ---")
        if len(rules_df) == 0:
            print("(Tidak ada rule yang ditemukan)")
            return
        for i, row in enumerate(rules_df.itertuples(), start=1):
            antecedent = ", ".join(sorted(row.antecedents))
            consequent = ", ".join(sorted(row.consequents))
            print(f"{i}. [{antecedent}]  ->  [{consequent}]   "
                  f"(support={row.support:.2%}, confidence={row.confidence:.2%}, lift={row.lift:.2f})")

    print("\n=== BARANG YANG SERING DIBELI & SALING TERHUBUNG ===")
    if metode_dipilih in ("1", "3"):
        cetak_rules("Hasil Apriori (Baseline)", baseline_rules)
    if metode_dipilih in ("2", "3"):
        cetak_rules("Hasil ARM + GA (Optimasi)", ga_rules)


def tampilkan_kesimpulan(summary_df, convergence, metode_dipilih):
    """Ubah tabel perbandingan jadi kesimpulan naratif yang mudah dibaca."""
    print("\n=== KESIMPULAN OTOMATIS ===")

    if metode_dipilih != "3":
        print("(Kesimpulan perbandingan hanya tersedia kalau kedua metode dijalankan sekaligus - pilih opsi 3.)")
        return

    baseline = summary_df[summary_df["Metode"] == "Apriori (Baseline)"].iloc[0]
    ga = summary_df[summary_df["Metode"] == "ARM + GA (Optimasi)"].iloc[0]

    if baseline["Jumlah Rule"] == 0 or ga["Jumlah Rule"] == 0:
        print("Salah satu metode tidak menemukan rule sama sekali (kemungkinan threshold terlalu ketat).")
        print("Coba turunkan minimum support/confidence lalu jalankan ulang.")
        return

    # --- Waktu eksekusi ---
    if ga["Waktu (detik)"] < baseline["Waktu (detik)"]:
        selisih = baseline["Waktu (detik)"] - ga["Waktu (detik)"]
        print(f"- Waktu eksekusi: GA LEBIH CEPAT {selisih:.4f} detik dibanding Apriori.")
    else:
        selisih = ga["Waktu (detik)"] - baseline["Waktu (detik)"]
        kali = ga["Waktu (detik)"] / baseline["Waktu (detik)"] if baseline["Waktu (detik)"] > 0 else float("inf")
        print(f"- Waktu eksekusi: GA LEBIH LAMBAT {selisih:.4f} detik (~{kali:.1f}x) dibanding Apriori. "
              f"Wajar pada dataset berukuran kecil-menengah karena Apriori masih sanggup menelusuri exhaustive.")

    # --- Jumlah rule ---
    if ga["Jumlah Rule"] < baseline["Jumlah Rule"]:
        print(f"- Jumlah rule: GA menemukan lebih SEDIKIT rule ({int(ga['Jumlah Rule'])} vs {int(baseline['Jumlah Rule'])}), "
              f"karena GA tidak menelusuri seluruh kombinasi seperti Apriori, hanya yang berhasil ditemukan lewat evolusi.")
    elif ga["Jumlah Rule"] > baseline["Jumlah Rule"]:
        print(f"- Jumlah rule: GA menemukan lebih BANYAK rule ({int(ga['Jumlah Rule'])} vs {int(baseline['Jumlah Rule'])}).")
    else:
        print(f"- Jumlah rule: SAMA BANYAK ({int(ga['Jumlah Rule'])} rule) di kedua metode.")

    # --- Kualitas rule (confidence & lift) ---
    if ga["Rata-rata Confidence"] > baseline["Rata-rata Confidence"]:
        print(f"- Kualitas rule (confidence): GA LEBIH BAIK, rata-rata {ga['Rata-rata Confidence']:.2%} "
              f"vs {baseline['Rata-rata Confidence']:.2%} pada Apriori.")
    else:
        print(f"- Kualitas rule (confidence): Apriori LEBIH BAIK, rata-rata {baseline['Rata-rata Confidence']:.2%} "
              f"vs {ga['Rata-rata Confidence']:.2%} pada GA.")

    if ga["Rata-rata Lift"] > baseline["Rata-rata Lift"]:
        print(f"- Kekuatan korelasi (lift): GA LEBIH BAIK, rata-rata {ga['Rata-rata Lift']:.2f} vs {baseline['Rata-rata Lift']:.2f}.")
    else:
        print(f"- Kekuatan korelasi (lift): Apriori LEBIH BAIK, rata-rata {baseline['Rata-rata Lift']:.2f} vs {ga['Rata-rata Lift']:.2f}.")

    if abs(ga["Confidence Maksimum"] - baseline["Confidence Maksimum"]) < 1e-6:
        print(f"- Rule terbaik: KEDUANYA menemukan rule dengan confidence maksimum yang SAMA ({ga['Confidence Maksimum']:.2%}), "
              f"membuktikan GA berhasil menemukan solusi optimal tanpa exhaustive search.")
    elif ga["Confidence Maksimum"] > baseline["Confidence Maksimum"]:
        print(f"- Rule terbaik: GA menemukan rule dengan confidence maksimum lebih TINGGI ({ga['Confidence Maksimum']:.2%} "
              f"vs {baseline['Confidence Maksimum']:.2%}).")
    else:
        print(f"- Rule terbaik: Apriori menemukan rule dengan confidence maksimum lebih tinggi ({baseline['Confidence Maksimum']:.2%} "
              f"vs {ga['Confidence Maksimum']:.2%}).")

    # --- Progres konvergensi GA ---
    if len(convergence) >= 2:
        fitness_awal, fitness_akhir = convergence[0], convergence[-1]
        if fitness_awal > 0:
            peningkatan = (fitness_akhir - fitness_awal) / fitness_awal * 100
            print(f"- Konvergensi GA: fitness terbaik naik dari {fitness_awal:.4f} (generasi 1) menjadi "
                  f"{fitness_akhir:.4f} (generasi {len(convergence)}), peningkatan {peningkatan:.1f}%, "
                  f"membuktikan proses evolusi benar-benar belajar, bukan menebak acak.")

    # --- Verdict ringkas ---
    print("\nRingkasan: ", end="")
    if ga["Rata-rata Confidence"] >= baseline["Rata-rata Confidence"] and ga["Waktu (detik)"] <= baseline["Waktu (detik)"]:
        print("GA unggul di kualitas rule maupun kecepatan - optimasi berhasil sepenuhnya.")
    elif ga["Rata-rata Confidence"] >= baseline["Rata-rata Confidence"]:
        print("GA unggul di kualitas rule (confidence lebih tinggi), tapi lebih lambat dari Apriori pada dataset "
              "sebesar ini. Keunggulan kecepatan GA baru akan terlihat pada dataset yang jauh lebih besar.")
    else:
        print("Pada percobaan ini Apriori masih lebih unggul secara umum. Coba sesuaikan parameter GA "
              "(populasi/generasi lebih besar, mutation rate lebih kecil) untuk hasil yang lebih optimal.")


def jalankan_analisis():
    """Satu siklus penuh: pilih metode -> input parameter manual -> jalankan -> simpan hasil."""
    global MIN_SUPPORT, MIN_CONFIDENCE, POP_SIZE, N_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE

    metode_dipilih = pilih_metode()
    MIN_SUPPORT, MIN_CONFIDENCE = atur_threshold_arm()

    baseline_rules, baseline_time = pd.DataFrame(), 0.0
    ga_rules, ga_time, convergence = pd.DataFrame(), 0.0, []

    if metode_dipilih in ("1", "3"):
        print("\n=== Menjalankan Apriori (baseline) ===")
        baseline_rules, baseline_time = run_apriori_baseline()
        print(f"Selesai dalam {baseline_time:.4f} detik | {len(baseline_rules)} rule ditemukan")

    if metode_dipilih in ("2", "3"):
        POP_SIZE, N_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE = atur_parameter_ga()
        print("\n=== Menjalankan ARM + Algoritma Genetika (optimasi) ===")
        ga_rules, ga_time, convergence = run_ga_arm()
        print(f"Selesai dalam {ga_time:.4f} detik | {len(ga_rules)} rule ditemukan")

    rows_summary = []
    if metode_dipilih in ("1", "3"):
        rows_summary.append(summarize(baseline_rules, baseline_time, "Apriori (Baseline)"))
    if metode_dipilih in ("2", "3"):
        rows_summary.append(summarize(ga_rules, ga_time, "ARM + GA (Optimasi)"))
    summary_df = pd.DataFrame(rows_summary)

    print("\n=== TABEL PERBANDINGAN ===")
    print(summary_df.to_string(index=False))

    tampilkan_daftar_rule(baseline_rules, ga_rules, metode_dipilih)
    tampilkan_kesimpulan(summary_df, convergence, metode_dipilih)

    baseline_readable = rules_to_readable(baseline_rules)
    ga_readable = rules_to_readable(ga_rules)

    OUTPUT_PATH = os.path.join(SCRIPT_DIR, "hasil_perbandingan_arm_ga.xlsx")
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Ringkasan_Perbandingan", index=False)
        if metode_dipilih in ("1", "3"):
            baseline_readable.to_excel(writer, sheet_name="Rule_Apriori_Baseline", index=False)
        if metode_dipilih in ("2", "3"):
            ga_readable.to_excel(writer, sheet_name="Rule_ARM_GA", index=False)
            pd.DataFrame({"Generasi": range(1, len(convergence) + 1),
                          "Best_Fitness": convergence}).to_excel(writer, sheet_name="Konvergensi_GA", index=False)

    print(f"\nHasil disimpan ke: {OUTPUT_PATH}")

    return baseline_rules, ga_rules, metode_dipilih


# =====================================================================
# 6. MENU UTAMA (LOOP) — jalan terus sampai user pilih Keluar
# =====================================================================
def menu_utama():
    while True:
        baseline_rules, ga_rules, metode_dipilih = jalankan_analisis()

        # Menu pasca-run: bukan langsung selesai, user pilih mau ngapain lagi
        while True:
            print("\n=== APA YANG INGIN DILAKUKAN SELANJUTNYA? ===")
            print("1. Lihat rekomendasi produk dari rule yang ditemukan")
            print("2. Jalankan ulang analisis (ganti metode/parameter)")
            print("3. Keluar")
            pilihan = input("Pilih (1/2/3): ").strip()

            if pilihan == "1":
                tampilkan_rekomendasi(baseline_rules, ga_rules, metode_dipilih)
                # tetap di menu ini setelah menampilkan rekomendasi
            elif pilihan == "2":
                break  # keluar dari menu pasca-run -> balik jalankan_analisis() lagi
            elif pilihan == "3":
                print("\nTerima kasih, program selesai.")
                return
            else:
                print("Input tidak dikenali, coba lagi.")


if __name__ == "__main__":
    menu_utama()