

# 欠損値補完手法の比較とモデル性能・内部構造の分析（HMEQ）

## 1. プロジェクト概要

本プロジェクトでは、**欠損値補完（Imputation）手法の違い**が、機械学習モデルの性能（特に **Recall**）にどのような影響を与えるかを比較・分析しました。
また、「スコアが上がった／下がった」で終わらず、**モデル内部（係数・特徴量重要度・サポートベクトル）**を確認することで、なぜその結果になったのかを説明できる形にまとめています。

### 取り組んだ主題

* 欠損値補完方法の違いにより **同じモデル内で性能はどう変わるか**
* 欠損値補完方法の違いにより **モデル間で性能の傾向はどう変わるか**
* なぜその補完方法が効いたのかを **モデル内部から説明できるか**

---

## 2. 背景・目的（Why）

就職活動のポートフォリオとして、「特徴量エンジニアリング」や「モデル構築」だけでなく、実務で頻出する **欠損値処理の設計**にフォーカスしました。

特に信用リスク（default予測）の文脈では、

* **defaultを見逃さないこと（Recall重視）**
* 欠損が「単なる欠測」ではなく **行動・属性のシグナル**になりうること

が重要です。

そのため本プロジェクトでは評価指標として **Recall** を主指標とし、補完方法とモデルの相性を検証しました。

---

## 3. 使用データ

* データセット：`hmeq.csv`（信用リスク関連データ）
* 目的変数：`BAD`（1 = default / 0 = non-default）
* 説明変数：数値＋カテゴリ（例：`LOAN`, `DEBTINC`, `REASON`, `JOB` など）

---

## 4. 全体の流れ（Pipeline 概要）

本プロジェクトは大きく以下の流れです。

1. **欠損パターンの可視化**

   * missingno による matrix / heatmap
2. **欠損がランダムかどうか（依存性）の簡易検定**

   * 欠損群 vs 観測群の平均差を t 検定で確認（MAR/MCARの示唆）
3. **欠損の多すぎる行の削除**

   * 非欠損が一定数未満の行を drop
4. **数値・カテゴリを分離して欠損補完**

   * 数値：平均／中央値／中央値＋欠損フラグ／MICE
   * カテゴリ：`Missing` で補完
5. **補完後分布の確認**

   * 観測値 vs 補完値のヒストグラム比較
6. **モデル別に外れ値処理を設計**

   * LR / SVM / DT で処理を変える（距離に敏感かどうか等）
7. **ColumnTransformer + Pipeline で前処理一体化**

   * カテゴリ：OneHotEncoder
   * 数値：StandardScaler（DTのみpass）
8. **StratifiedKFold で5分割CV**

   * `cross_validate(scoring="recall")`
   * mean と std を記録（性能＋安定性）
9. **モデル内部の分析**

   * Decision Tree：`feature_importances_`、どの深さで使われたか
   * Logistic Regression：係数、odds ratio、median_flag vs mice比較
   * SVM：サポートベクトル数、SV内default率、SV内欠損フラグ率

---

## 5. 比較した欠損補完方法

本プロジェクトで比較した補完は以下の4種類です。

### 5.1 mean（平均補完）

* `LOAN, MORTDUE, VALUE, CLAGE, CLNO, DEBTINC`：平均
* `YOJ, DEROG, DELINQ, NINQ`：0（回数系で欠損は0の可能性があるという設計）

### 5.2 median（中央値補完）

* 上記と同様の列設計で中央値補完

### 5.3 median_flag（中央値補完 + 欠損フラグ）

* `YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC` に対して
  `*_MISSING`（0/1）を追加
* その後に中央値/0で補完

### 5.4 MICE（IterativeImputer）

* `IterativeImputer(estimator=BayesianRidge())`
* 数値変数のみMICE適用 → カテゴリは別途結合
* ※ MICEには欠損フラグは付与しない設計

---

## 6. 使用モデル

* Logistic Regression（class_weight="balanced"）
* SVM（RBF kernel, class_weight="balanced", probability=True）
* Decision Tree

※本プロジェクトの目的は「欠損補完の比較」にあるため、過度なハイパーパラメータ探索はせず、再現性のあるベース構成で比較しています。

---

## 7. 前処理（ColumnTransformer / Pipeline）

### 7.1 カテゴリ変数

* `REASON`, `JOB` を One-Hot Encoding
* `drop="first"`（多重共線性対策）
* `handle_unknown="ignore"`

### 7.2 数値変数

* LR/SVM：StandardScaler
* DT：スケーリングなし（passthrough）

---

## 8. 外れ値処理（モデル別）

### 8.1 Logistic Regression 用

* `CLAGE`：99%キャップ
* `YOJ, CLNO, LOAN, MORTDUE, VALUE, DEBTINC`：log1p

### 8.2 SVM 用（距離に敏感なので強め）

* 金額系：99%キャップ + log1p
* `DELINQ`: upper=5, `DEROG`: upper=3, `NINQ`: upper=6（回数系は情報を残しつつキャップ）
* `YOJ, CLNO, DEBTINC`：log1p
* `CLAGE`：99%キャップのみ

### 8.3 Decision Tree 用

* 原則不要（閾値分割なので距離に依存しない）
* ただし `CLAGE` の非現実的極端値のみ99%キャップ

---

## 9. 評価方法

* 交差検証：StratifiedKFold（5分割, shuffle=True, random_state=42）
* 指標：**Recall**
* 出力：`recall_mean`, `recall_std`

---

## 10. 結果サマリ（Recall）

もちろんです。
では、**あなたが出してくれた `df_results_all` の数値をそのまま反映した README 用 Results セクション**を作ります。
これは **コピペで完成版**です。

---

## Results（交差検証結果）

本節では、各モデル × 欠損補完方法の **Recall（平均）および標準偏差** を示す。
評価は **StratifiedKFold（5分割）** による交差検証で行い、
**default を見逃さないこと（Recall）** を最重要指標とした。

---

### Recall（mean ± std）

> ※ Recall は「実際に default した顧客のうち、正しく default と予測できた割合」を表す。

| Model               | Imputation Method         | Recall (mean) | Recall (std) |
| ------------------- | ------------------------- | ------------- | ------------ |
| **SVM**             | **Median + Missing Flag** | **0.8269**    | **0.0350**   |
| SVM                 | MICE                      | 0.8079        | 0.0190       |
| Logistic Regression | Median + Missing Flag     | 0.7727        | 0.0625       |
| SVM                 | Median                    | 0.7520        | 0.0126       |
| SVM                 | Mean                      | 0.7476        | 0.0034       |
| Logistic Regression | MICE                      | 0.7339        | 0.0365       |
| Decision Tree       | MICE                      | 0.7106        | 0.0176       |
| Logistic Regression | Mean                      | 0.6762        | 0.0379       |
| Logistic Regression | Median                    | 0.6762        | 0.0354       |
| Decision Tree       | Mean                      | 0.6512        | 0.0216       |
| Decision Tree       | Median + Missing Flag     | 0.6486        | 0.0404       |
| Decision Tree       | Median                    | 0.6477        | 0.0244       |

---

### Best Result（最良結果）

* **Best Model**：SVM
* **Best Imputation**：Median + Missing Flag
* **Highest Recall**：**0.8269**

本データセットにおいては、
**欠損を明示的な特徴量として扱う設計（欠損フラグ）**が
distance-based なモデルである SVM と特に相性が良いことが確認された。

---

### 安定性（標準偏差）について

* `Recall (std)` が小さい手法ほど、
  **データ分割に依存しにくい安定した性能**を示している
* SVM + Median は std が非常に小さい一方、
  Median + Missing Flag は Recall 向上と引き換えに分散がやや大きい
  → **高リコール・やや高分散**というトレードオフが存在

---

### 結果の要点（1分要約）

* **最高 Recall**：SVM × Median + Missing Flag（0.8269）
* **LR / SVM**：欠損フラグが有効
* **DT**：MICE が最良
* 欠損補完の最適解は **モデル依存**

---

## 11. モデル内部の分析（なぜそうなったか）

### 11.1 Logistic Regression（係数・odds ratio）

* `median_flag` では欠損フラグがロジット（対数オッズ）に直接加算され、欠損そのもののリスクを表現できる
* 例：`DEBTINC_MISSING` の係数・オッズ比から、欠損が強いリスクシグナルになっていることを確認
* `mice` では欠損フラグがないため、連続値（例：DEBTINC）が欠損リスクも含めて引き受ける構造になる

### 11.2 Decision Tree（特徴量重要度・分割構造）

* 決定木は分割ごとに最も不純度を下げる特徴量のみを使うため、欠損フラグが全て効くとは限らない
* `median` のように値が一点集中すると分割の自由度が下がりやすい
* `mice` は分布が滑らかになるため深い層でも分割しやすく、DTでは相性が良い傾向

### 11.3 SVM（サポートベクトル分析）

SVMは「係数」や「重要度」で寄与を直接分解しにくいので、以下を分析しました。

* サポートベクトル数・比率
* サポートベクトル内 default 率（難しいサンプルが境界に集まっているか）
* `median_flag` の場合、SV内の欠損フラグ率（どの欠損が境界形成に強く関与しているか）

その結果、欠損（特に `DEBTINC` 欠損）が境界形成に強く関与していることが確認できました。

---

## 12. リポジトリ構成（例）

推奨構成（今後GitHubに載せる場合）

```
.
├── README.md
├── notebooks/
│   └── final_project_portfolio.ipynb
├── data/
    └── hmeq.csv  

```

---

## 13. 実行方法（Colab想定）

### 13.1 前提

* Google Colab 環境
* Google Drive に `hmeq.csv` を配置
* ノートブック内でDriveをマウントして読み込み

### 13.2 依存ライブラリ

主な使用ライブラリ：

* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* missingno
* scipy

Colabでは冒頭で `missingno` をインストールしています。

---

## 14. 今後の改善案（Optional / 発展）

* ハイパーパラメータ探索（ただし目的が「補完比較」なので優先度低）
* `RobustScaler` や `QuantileTransformer` の比較（特にSVM）
* 欠損機構（MCAR/MAR/NMAR）のより厳密な推定
* SHAP/LIME等の説明手法

---

## 15. まとめ

* 欠損補完の最適解は「1つ」ではなく、**モデル特性と欠損の意味に依存する**
* LR/SVMでは欠損フラグが有効になりやすい
* DTでは連続値の分布を保つMICEが有効になりやすい
* 「なぜそうなったか」をモデル内部（係数/重要度/SV）から説明できる形にした

