"""
test_api.py — DDI Prediction API Test Suite
============================================
Tests all three endpoints: /health, /predict, /drugs
Run with: python test_api.py
Make sure the API is running at http://localhost:8000 first.
"""

import requests
import json
import time
import sys
from typing import Optional

BASE_URL = "http://localhost:8000/"

# ─── ANSI colors for terminal output ───────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ─── Test result tracking ───────────────────────────────────────────────────────
results = {"passed": 0, "failed": 0, "warnings": 0}

def passed(msg: str):
    results["passed"] += 1
    print(f"  {GREEN}✓ PASS{RESET}  {msg}")

def failed(msg: str, detail: Optional[str] = None):
    results["failed"] += 1
    print(f"  {RED}✗ FAIL{RESET}  {msg}")
    if detail:
        print(f"         {DIM}{detail}{RESET}")

def warning(msg: str):
    results["warnings"] += 1
    print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")

def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")

def check_api_running():
    """Confirm API is reachable before running any tests."""
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ══════════════════════════════════════════════════════════════════════════════

def test_root():
    section("1. ROOT ENDPOINT  GET /")
    r = requests.get(f"{BASE_URL}/", timeout=5)

    if r.status_code == 200:
        passed(f"Status code 200")
    else:
        failed(f"Expected 200, got {r.status_code}")
    print("Status:", r.status_code)
    print("Content-Type:", r.headers.get("Content-Type"))
    print("Response:")
    print(r.text)
    data = r.json()
    for key in ["message", "docs", "endpoints"]:
        if key in data:
            passed(f"Response contains '{key}' field")
        else:
            failed(f"Missing '{key}' field in response", str(data))


def test_health():
    section("2. HEALTH ENDPOINT  GET /health")
    r = requests.get(f"{BASE_URL}/health", timeout=5)

    if r.status_code == 200:
        passed("Status code 200")
    else:
        failed(f"Expected 200, got {r.status_code}")
        return

    data = r.json()
    print(f"\n  {DIM}Raw response: {json.dumps(data, indent=4)}{RESET}\n")

    # Required fields
    for key in ["status", "model", "drugs_in_graph", "auroc", "threshold"]:
        if key in data:
            passed(f"Field '{key}' present: {CYAN}{data[key]}{RESET}")
        else:
            failed(f"Missing field '{key}'")

    # Validate values
    if data.get("status") == "healthy":
        passed("Status is 'healthy'")
    else:
        failed(f"Status should be 'healthy', got '{data.get('status')}'")

    if isinstance(data.get("drugs_in_graph"), int) and data["drugs_in_graph"] > 1000:
        passed(f"Drug count looks valid: {data['drugs_in_graph']}")
    else:
        warning(f"Drug count seems low: {data.get('drugs_in_graph')}")

    if isinstance(data.get("auroc"), float) and data["auroc"] > 0.90:
        passed(f"AUROC looks valid: {data['auroc']}")
    else:
        warning(f"AUROC seems low or missing: {data.get('auroc')}")


def test_drugs_endpoint():
    section("3. DRUGS ENDPOINT  GET /drugs")
    r = requests.get(f"{BASE_URL}/drugs", timeout=10)

    if r.status_code == 200:
        passed("Status code 200")
    else:
        failed(f"Expected 200, got {r.status_code}")
        return

    data = r.json()

    if "total" in data:
        passed(f"'total' field present: {CYAN}{data['total']}{RESET}")
    else:
        failed("Missing 'total' field")

    if "drugs" in data and isinstance(data["drugs"], list):
        passed(f"'drugs' field is a list with {len(data['drugs'])} entries")
    else:
        failed("'drugs' field missing or not a list")
        return

    if len(data["drugs"]) > 1000:
        passed(f"Drug list is substantial ({len(data['drugs'])} drugs)")
    else:
        warning(f"Drug list seems short: {len(data['drugs'])} drugs")

    # Check list is sorted
    if data["drugs"] == sorted(data["drugs"]):
        passed("Drug list is alphabetically sorted")
    else:
        warning("Drug list is not sorted — consider sorting in /drugs endpoint")

    # Print a sample
    sample = data["drugs"][:5]
    print(f"\n  {DIM}Sample drugs: {sample}{RESET}\n")

    return data["drugs"]  # return for use in predict tests


def test_predict_valid_pair(drug_a: str, drug_b: str, expect_risk: Optional[str] = None):
    """Test a known drug pair and validate response structure."""
    payload = {"drug_a": drug_a, "drug_b": drug_b}
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)

    label = f"{drug_a} + {drug_b}"

    if r.status_code == 200:
        passed(f"[{label}] Status code 200")
    else:
        failed(f"[{label}] Expected 200, got {r.status_code}", r.text)
        return

    data = r.json()
    print(f"\n  {DIM}Response: {json.dumps(data, indent=4)}{RESET}\n")

    # Required fields
    for key in ["drug_a", "drug_b", "interaction_probability",
                "risk_level", "both_drugs_known", "disclaimer"]:
        if key in data:
            passed(f"[{label}] Field '{key}' present")
        else:
            failed(f"[{label}] Missing field '{key}'")

    # Probability is a valid float in [0, 1]
    prob = data.get("interaction_probability")
    if isinstance(prob, float) and 0.0 <= prob <= 1.0:
        passed(f"[{label}] Probability is valid float: {CYAN}{prob:.4f}{RESET}")
    else:
        failed(f"[{label}] Probability invalid: {prob}")

    # Risk level is one of the expected values
    risk = data.get("risk_level")
    if risk in ["HIGH", "MEDIUM", "LOW"]:
        color = RED if risk == "HIGH" else YELLOW if risk == "MEDIUM" else GREEN
        passed(f"[{label}] Risk level valid: {color}{risk}{RESET}")
    else:
        failed(f"[{label}] Invalid risk level: {risk}")

    # Optional: check expected risk
    if expect_risk and risk != expect_risk:
        warning(f"[{label}] Expected {expect_risk}, got {risk} — model may differ")

    # both_drugs_known should be True for valid pairs
    if data.get("both_drugs_known") is True:
        passed(f"[{label}] both_drugs_known = True")
    else:
        warning(f"[{label}] both_drugs_known = False — drug may not be in graph")

    # Disclaimer must not be empty
    if data.get("disclaimer") and len(data["disclaimer"]) > 10:
        passed(f"[{label}] Disclaimer present")
    else:
        failed(f"[{label}] Disclaimer missing or too short")


def test_predict_known_interaction():
    section("4. PREDICT — KNOWN INTERACTING PAIRS")
    # These are pharmacologically known high-risk pairs
    known_pairs = [
        ("Warfarin", "Deferasirox"),
        ("Methotrexate", "Ibuprofen"),
        ("Fluconazole", "Warfarin"),
    ]
    for drug_a, drug_b in known_pairs:
        test_predict_valid_pair(drug_a, drug_b, expect_risk="HIGH")


def test_predict_case_insensitive():
    section("5. PREDICT — CASE INSENSITIVITY")
    # All three should return the same result
    pairs = [
        ("warfarin", "deferasirox"),
        ("WARFARIN", "DEFERASIROX"),
        ("Warfarin", "Deferasirox"),
    ]
    probs = []
    for drug_a, drug_b in pairs:
        r = requests.post(
            f"{BASE_URL}/predict",
            json={"drug_a": drug_a, "drug_b": drug_b},
            timeout=10
        )
        if r.status_code == 200:
            prob = r.json().get("interaction_probability")
            probs.append(prob)
            passed(f"'{drug_a}' + '{drug_b}' → prob={prob}")
        else:
            failed(f"'{drug_a}' + '{drug_b}' → status {r.status_code}")

    if len(set(probs)) == 1:
        passed("All case variants return identical probability ✓")
    else:
        failed("Case variants return different probabilities — case normalization broken", str(probs))


def test_predict_symmetry():
    section("6. PREDICT — SYMMETRY  (A,B) == (B,A)")
    drug_a, drug_b = "Warfarin", "Deferasirox"

    r1 = requests.post(f"{BASE_URL}/predict", json={"drug_a": drug_a, "drug_b": drug_b}, timeout=10)
    r2 = requests.post(f"{BASE_URL}/predict", json={"drug_a": drug_b, "drug_b": drug_a}, timeout=10)

    if r1.status_code == 200 and r2.status_code == 200:
        prob1 = r1.json().get("interaction_probability")
        prob2 = r2.json().get("interaction_probability")
        print(f"\n  {DIM}({drug_a}, {drug_b}) → {prob1}{RESET}")
        print(f"  {DIM}({drug_b}, {drug_a}) → {prob2}{RESET}\n")

        if prob1 == prob2:
            passed("Prediction is symmetric — (A,B) == (B,A) ✓")
        else:
            # Floating point might cause tiny differences — check within tolerance
            if abs(prob1 - prob2) < 1e-6:
                passed("Prediction is symmetric within floating point tolerance ✓")
            else:
                failed(
                    "Prediction is NOT symmetric — edge features are order-dependent",
                    f"({drug_a},{drug_b})={prob1}  vs  ({drug_b},{drug_a})={prob2}"
                )
    else:
        failed("One or both requests failed", f"{r1.status_code}, {r2.status_code}")


def test_predict_unknown_drug():
    section("7. PREDICT — UNKNOWN DRUG HANDLING")

    # One unknown drug
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"drug_a": "Unicornazole", "drug_b": "Warfarin"},
        timeout=10
    )

    if r.status_code == 200:
        data = r.json()
        if data.get("both_drugs_known") is False:
            passed("Unknown drug returns both_drugs_known=False")
        else:
            warning("Unknown drug didn't set both_drugs_known=False")

        if data.get("risk_level") == "UNKNOWN":
            passed("Risk level is 'UNKNOWN' for unknown drug")
        else:
            warning(f"Expected 'UNKNOWN' risk, got '{data.get('risk_level')}'")

        if data.get("interaction_probability") is None:
            passed("Probability is None for unknown drug (correct)")
        else:
            warning(f"Probability should be None, got {data.get('interaction_probability')}")

        print(f"\n  {DIM}Response: {json.dumps(data, indent=4)}{RESET}\n")
    else:
        failed(f"Unknown drug returned status {r.status_code} instead of 200")

    # Both unknown
    r2 = requests.post(
        f"{BASE_URL}/predict",
        json={"drug_a": "Fakecillin", "drug_b": "Placebostat"},
        timeout=10
    )
    if r2.status_code == 200:
        passed("Both-unknown request returns 200 (graceful, not crash)")
    else:
        failed(f"Both-unknown request crashed with {r2.status_code}")


def test_predict_same_drug():
    section("8. PREDICT — SAME DRUG BOTH FIELDS")
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"drug_a": "Warfarin", "drug_b": "Warfarin"},
        timeout=10
    )

    if r.status_code == 400:
        passed("Same drug returns 400 Bad Request ✓")
        print(f"  {DIM}Detail: {r.json().get('detail')}{RESET}")
    elif r.status_code == 200:
        warning("Same drug pair returned 200 — consider adding validation to reject this")
    else:
        warning(f"Same drug returned {r.status_code} — unexpected")


def test_predict_empty_fields():
    section("9. PREDICT — MISSING / EMPTY FIELDS")

    bad_payloads = [
        ({}, "Empty payload"),
        ({"drug_a": "Warfarin"}, "Missing drug_b"),
        ({"drug_b": "Aspirin"}, "Missing drug_a"),
        ({"drug_a": "", "drug_b": "Aspirin"}, "Empty drug_a string"),
        ({"drug_a": "Warfarin", "drug_b": ""}, "Empty drug_b string"),
    ]

    for payload, label in bad_payloads:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        if r.status_code in [400, 422]:
            passed(f"[{label}] Correctly rejected with {r.status_code}")
        elif r.status_code == 200:
            warning(f"[{label}] Returned 200 — consider stricter input validation")
        else:
            failed(f"[{label}] Unexpected status {r.status_code}")


def test_response_time():
    section("10. PERFORMANCE — RESPONSE TIME")

    pairs = [
        ("Warfarin", "Deferasirox"),
        ("Methotrexate", "Ibuprofen"),
        ("Fluconazole", "Warfarin"),
        ("Digoxin", "Amiodarone"),
        ("Lithium", "Ibuprofen"),
    ]

    times = []
    for drug_a, drug_b in pairs:
        start = time.time()
        r = requests.post(
            f"{BASE_URL}/predict",
            json={"drug_a": drug_a, "drug_b": drug_b},
            timeout=10
        )
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        status = "✓" if r.status_code == 200 else "✗"
        print(f"  {DIM}{status} {drug_a} + {drug_b}: {elapsed:.1f}ms{RESET}")

    avg_ms = sum(times) / len(times)
    max_ms = max(times)

    print()
    if avg_ms < 200:
        passed(f"Average response time: {CYAN}{avg_ms:.1f}ms{RESET} (target < 200ms)")
    elif avg_ms < 500:
        warning(f"Average response time: {avg_ms:.1f}ms (acceptable but could be faster)")
    else:
        failed(f"Average response time: {avg_ms:.1f}ms (too slow — check model loading)")

    if max_ms < 500:
        passed(f"Max response time: {CYAN}{max_ms:.1f}ms{RESET}")
    else:
        warning(f"Max response time: {max_ms:.1f}ms — one request was slow")


def test_probability_range():
    section("11. PROBABILITY RANGE — SANITY CHECK")

    # Sample 10 different pairs and confirm all probs in [0,1]
    pairs = [
        ("Warfarin", "Deferasirox"),
        ("Metformin", "Ibuprofen"),
        ("Atorvastatin", "Clarithromycin"),
        ("Digoxin", "Amiodarone"),
        ("Lithium", "Ibuprofen"),
        ("Fluconazole", "Simvastatin"),
        ("Clopidogrel", "Omeprazole"),
        ("Methotrexate", "Naproxen"),
        ("Cyclosporine", "Ketoconazole"),
        ("Theophylline", "Ciprofloxacin"),
    ]

    probs = []
    for drug_a, drug_b in pairs:
        r = requests.post(
            f"{BASE_URL}/predict",
            json={"drug_a": drug_a, "drug_b": drug_b},
            timeout=10
        )
        if r.status_code == 200:
            prob = r.json().get("interaction_probability")
            if prob is not None:
                probs.append(prob)
                risk = r.json().get("risk_level")
                color = RED if risk=="HIGH" else YELLOW if risk=="MEDIUM" else GREEN
                print(f"  {DIM}{drug_a} + {drug_b}: {prob:.4f} → {color}{risk}{RESET}")

    print()
    if all(0.0 <= p <= 1.0 for p in probs):
        passed(f"All {len(probs)} probabilities in valid range [0, 1] ✓")
    else:
        failed("Some probabilities outside [0, 1]", str(probs))

    risk_counts = {}
    for drug_a, drug_b in pairs:
        r = requests.post(
            f"{BASE_URL}/predict",
            json={"drug_a": drug_a, "drug_b": drug_b},
            timeout=10
        )
        if r.status_code == 200:
            risk = r.json().get("risk_level", "UNKNOWN")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

    print(f"\n  {DIM}Risk distribution across test pairs: {risk_counts}{RESET}")
    if len(risk_counts) > 1:
        passed("Model predicts multiple risk levels (not stuck on one class)")
    else:
        warning("Model only predicted one risk level — check threshold settings")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    total = results["passed"] + results["failed"]
    section("TEST SUMMARY")
    print(f"  {GREEN}Passed:  {results['passed']}{RESET}")
    print(f"  {RED}Failed:  {results['failed']}{RESET}")
    print(f"  {YELLOW}Warnings:{results['warnings']}{RESET}")
    print(f"  Total:   {total}\n")

    if results["failed"] == 0:
        print(f"  {GREEN}{BOLD}ALL TESTS PASSED ✓{RESET}\n")
    else:
        print(f"  {RED}{BOLD}{results['failed']} TEST(S) FAILED ✗{RESET}\n")


if __name__ == "__main__":
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  DDI PREDICTION API — TEST SUITE{RESET}")
    print(f"{BOLD}  Target: {BASE_URL}{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    # Confirm API is reachable before running anything
    print(f"\n{DIM}Checking API availability...{RESET}")
    if not check_api_running():
        print(f"\n{RED}{BOLD}ERROR: Cannot reach API at {BASE_URL}{RESET}")
        print(f"{YELLOW}Make sure the server is running:{RESET}")
        print(f"  cd ddi_api")
        print(f"  uvicorn app.main:app --reload --port 8000\n")
        sys.exit(1)

    print(f"{GREEN}API is reachable ✓{RESET}")

    # Run all tests
    test_root()
    test_health()
    test_drugs_endpoint()
    test_predict_known_interaction()
    test_predict_case_insensitive()
    test_predict_symmetry()
    test_predict_unknown_drug()
    test_predict_same_drug()
    test_predict_empty_fields()
    test_response_time()
    test_probability_range()

    print_summary()