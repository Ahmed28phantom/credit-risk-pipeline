from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

NUMERIC_FEATURES = [
    "duration",
    "amount",
    "installment_rate",
    "present_residence",
    "age",
    "number_credits",
    "people_liable",
]

CATEGORICAL_FEATURES = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer that scales numeric features
    and one-hot encodes categorical features.
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def build_pipeline(use_smote: bool = False) -> Pipeline:
    """
    Combines the ColumnTransformer preprocessor with a
    Logistic Regression model. If use_smote=True, inserts a
    SMOTE step after preprocessing to rebalance classes
    during training only (skipped automatically at predict time).
    """
    preprocessor = build_preprocessor()
    model = LogisticRegression(max_iter=1000, random_state=7)

    steps = [("preprocessor", preprocessor)]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=7)))

    steps.append(("classifier", model))

    return Pipeline(steps=steps)