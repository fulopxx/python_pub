"""Adversarial credit-reporting hypotheses, not allegations or fraud probabilities."""
from __future__ import annotations

import re
from datetime import date
from macro_ai_credit_monitor import number

PATTERNS = [
    ("MODIFICATIONS", "Átstrukturálás / lejárat-hosszabbítás",
     r"\b(forbearance|loan modifications?|maturity extensions?|extend.and.pretend|restructur(?:ed|ing))\b",
     "Fizetőképesség javult, vagy csak későbbre került az esedékesség?"),
    ("NONCASH_INTEREST", "Nem készpénzes kamat / PIK",
     r"\b(payment.in.kind|PIK interest|capitalized interest|interest capitalization)\b",
     "A bevétel mellé érkezik készpénz? A PIK önmagában nem szabálytalan."),
    ("RESERVE_MODEL", "Céltartalék / modell / feltételezés változása",
     r"\b(reserve releases?|qualitative adjustments?|model changes?|CECL|economic forecast assumptions?)\b",
     "Azonos portfólióra, NCO-ra és átvezetésekre bontott ACL roll-forward szükséges."),
    ("PORTFOLIO_TRANSFER", "Hitelértékesítés / átsorolás",
     r"\b(loan sales?|held.for.sale|reclassif(?:ied|ication)|securiti[sz]ation|transfer(?:red)? loans?)\b",
     "Csökkent a kockázat, vagy azonos kitettség más sorra került? Eladási veszteség és recourse kell."),
    ("OFF_BALANCE", "Mérlegen kívüli / közvetett kitettség",
     r"\b(off.balance.sheet|variable interest entit(?:y|ies)|unfunded commitments?|warehouse financing)\b",
     "Ki viseli a végső kockázatot? Közvetett + közvetlen kitettséget nem szabad duplázni."),
    ("CREDIT_MIGRATION", "Minősítésromlás / nonaccrual / késedelem",
     r"\b(nonaccrual|non.performing|criticized|classified loans?|past.due|delinquenc(?:y|ies))\b",
     "A minősítésromlás előbb jelenik meg, mint a veszteség? Azonos definíciójú idősor kell."),
]
AI_PATTERN = re.compile(r"\bAI\b|artificial intelligence|data cent(?:er|re)s?", re.I)
DISCLAIMER = ("Lehetséges késleltetési/cosmetic-reporting hipotézisek; "
              "nem bizonyított eltitkolás, nem csalásminősítés, nincs kalibrált valószínűség.")


def scan_disclosures(documents):
    """Keyword evidence only. Preserve context/negation and never infer sentiment."""
    hits = []
    for doc in documents:
        if doc.get("status") not in ("OK", "TEXT_EXTRACTED"):
            continue
        text = doc.get("text", "")
        for code, label, pattern, question in PATTERNS:
            matches = list(re.finditer(pattern, text, re.I))
            if not matches:
                continue
            match = matches[0]
            snippet = text[max(0, match.start()-160):min(len(text), match.end()+220)]
            hits.append(dict(
                ticker=doc["ticker"], category=code, label=label,
                status="TEXT_MENTION_NOT_ADVERSE_CONCLUSION",
                priority="MANUAL_REVIEW", probability=None,
                ai_context_present=bool(AI_PATTERN.search(snippet)),
                excerpt=snippet, question=question, source=doc.get("source"),
                accession=doc.get("accession"), accepted_at=doc.get("accepted_at"),
                document_quality=doc.get("document_quality", "UNKNOWN"),
                negation_or_boilerplate_not_resolved=True))
    return hits


def quantitative_checks(rows):
    """Only explicit comparable normalized quarters; absent data is NOT a pass.

    scope_id is a provider/manual normalization assertion for all listed metrics,
    not guessed from tag labels. Roll-forward only when all six terms share it.
    """
    numeric = ("npl_usd", "allowance_usd", "net_chargeoffs_usd", "provision_usd",
               "modified_loans_usd", "pik_interest_usd", "acl_other_changes_usd")
    rows = [dict(r, **{k: number(r.get(k)) for k in numeric}) for r in rows]
    out = []
    tickers = sorted({r["ticker"] for r in rows})
    for ticker in tickers:
        records = sorted((r for r in rows if r["ticker"] == ticker),
                         key=lambda r: r["period_end"])
        for index, current in enumerate(records):
            checks = []
            previous = records[index-1] if index else None
            comparable = False
            if previous is not None:
                delta = (date.fromisoformat(current["period_end"])-
                         date.fromisoformat(previous["period_end"])).days
                comparable = (70 <= delta <= 110 and bool(current.get("scope_id")) and
                              current.get("scope_id") == previous.get("scope_id") and
                              current.get("normalization_verified") is True and
                              previous.get("normalization_verified") is True)
            def missing_reasons(current_fields, previous_fields, flow=False):
                missing = []
                if previous is None:
                    missing.append("Nincs előző összehasonlítható negyedév.")
                elif not 70 <= (date.fromisoformat(current["period_end"])-
                                date.fromisoformat(previous["period_end"])).days <= 110:
                    missing.append("Hiányzik a közvetlenül előző negyedév; időszakrés van.")
                if not current.get("normalization_verified"):
                    missing.append("Aktuális negyedév: nincs jóváhagyott gépi/forrás szerinti normalizálás.")
                if previous and not previous.get("normalization_verified"):
                    missing.append(f"{previous['period_end']}: nincs normalizált adat.")
                if previous and (not current.get("scope_id") or
                                 current.get("scope_id") != previous.get("scope_id")):
                    missing.append("Portfólió/scope eltér vagy nincs megadva.")
                if flow and (current.get("flow_basis") != "QUARTER" or
                             previous and previous.get("flow_basis") != "QUARTER"):
                    missing.append("A flow nem igazolt negyedéves adat (YTD/éves nem használható).")
                for record, fields in ((current, current_fields), (previous, previous_fields)):
                    if record is None:
                        continue
                    for field in fields:
                        if number(record.get(field)) is None:
                            why = record.get("extraction_issues", {}).get(field)
                            missing.append(f"{record['period_end']}: {field} hiányzik"+
                                           (f" [{why}]" if why else ""))
                # Auto normalization checks only with stable per-metric definitions.
                if previous and (current.get("normalization_method") or previous.get("normalization_method")):
                    for field in set(current_fields) & set(previous_fields):
                        old = previous.get("metric_provenance", {}).get(field)
                        new = current.get("metric_provenance", {}).get(field)
                        if old and new and old.get("definition_id") != new.get("definition_id"):
                            missing.append(f"{field}: taxonómia/definíció változott, automatikus összevetés tiltva.")
                    for record, fields in ((current, current_fields), (previous, previous_fields)):
                        for field in fields:
                            if number(record.get(field)) is not None and not record.get("metric_provenance", {}).get(field):
                                missing.append(f"{record['period_end']}: {field} forrásproveniencia hiányzik.")
                return missing
            def check(code, label, required, fn, explanation):
                missing = missing_reasons(required, required,
                                          code in ("NCO_UP_PROVISION_DOWN", "PIK_UP"))
                if missing:
                    state, detail = "NOT_TESTABLE", " ".join(missing)
                else:
                    state = "REVIEW_FLAG" if fn() else "NO_TRIGGER"
                    detail = explanation
                checks.append(dict(code=code, label=label, status=state, explanation=detail,
                                   missing_reasons=missing, probability=None, scope_id=current.get("scope_id")))
            check("NPL_UP_ACL_DOWN", "NPL nő, ACL csökken",
                  ("npl_usd", "allowance_usd"),
                  lambda: current["npl_usd"] > previous["npl_usd"] and
                          current["allowance_usd"] < previous["allowance_usd"],
                  "Fedezetcsökkenés minősítésromlás mellett; mix/model/értékesítés is magyarázhatja.")
            check("NCO_UP_PROVISION_DOWN", "NCO nő, provision csökken",
                  ("net_chargeoffs_usd", "provision_usd"),
                  lambda: current["net_chargeoffs_usd"] > previous["net_chargeoffs_usd"] and
                          current["provision_usd"] < previous["provision_usd"],
                  "Utólagos veszteség vs előretekintő becslés eltér; nem automatikus alulképzés.")
            check("MODIFICATIONS_UP", "Módosított hitelek nőnek",
                  ("modified_loans_usd",),
                  lambda: current["modified_loans_usd"] > previous["modified_loans_usd"],
                  "Növekvő módosítási állomány; a számláló és definíció változatlansága ellenőrizendő.")
            check("PIK_UP", "Nem cash kamat nő",
                  ("pik_interest_usd",),
                  lambda: current["pik_interest_usd"] > previous["pik_interest_usd"],
                  "Nem cash komponens nő; szerződéses feltételek és cash collection külön szükséges.")
            fields = ("allowance_usd", "provision_usd", "net_chargeoffs_usd", "acl_other_changes_usd")
            missing = missing_reasons(fields, ("allowance_usd",), flow=True)
            if current.get("normalization_method") and not missing:
                # A roll-forward combines several flows: one filing and quarter.
                ev = current["metric_provenance"]
                accessions = {ev[k].get("accession") for k in fields}
                if None in accessions or len(accessions) != 1:
                    missing.append("ACL-mérleg: a komponensek nem ugyanazon filingból származnak.")
                starts = {ev[k].get("start") for k in fields if k != "allowance_usd"}
                if len(starts) != 1:
                    missing.append("ACL-mérleg: eltérő flow-kezdőidőpontok.")
            complete = not missing
            residual = (current["allowance_usd"] -
                        (previous["allowance_usd"]+current["provision_usd"]-
                         current["net_chargeoffs_usd"]+current["acl_other_changes_usd"])) if complete else None
            tolerance = max(1_000_000, abs(current["allowance_usd"])*.001) if complete else None
            if complete and current.get("normalization_method"):
                def uncertainty(evidence):
                    if evidence.get("components"):
                        return sum(uncertainty(e) for e in evidence["components"])
                    try:
                        decimals = int(evidence.get("decimals"))
                        return .5 * 10**(-decimals) if -12 <= decimals <= 12 else 0
                    except (ValueError, TypeError):
                        return 0
                rounding = sum(uncertainty(current["metric_provenance"][k]) for k in fields)
                rounding += uncertainty(previous["metric_provenance"]["allowance_usd"])
                tolerance = max(tolerance, rounding)
            checks.append(dict(code="ACL_ROLL_FORWARD", label="ACL roll-forward egyezés",
                               status="NOT_TESTABLE" if residual is None else
                               "REVIEW_FLAG" if abs(residual) > tolerance else "NO_TRIGGER",
                               residual_usd=residual, tolerance_usd=tolerance,
                               missing_reasons=missing,
                               explanation="ACL1=ACL0+provision−NCO+other. Az other nem alapértelmezett nulla.",
                               probability=None))
            out.append(dict(ticker=ticker, period_end=current["period_end"], checks=checks,
                            source=current.get("source"), comparable=comparable,
                            normalization_method=current.get("normalization_method"),
                            previous_period_end=previous["period_end"] if previous else None,
                            available_at=current.get("available_at")))
    return out


def evaluate_forensics(documents, credit_rows):
    hits = scan_disclosures(documents)
    checks = quantitative_checks(credit_rows)
    return dict(hypothesis_mode="ADVERSARIAL_NOT_PRESUMPTION_OF_GUILT",
                disclaimer=DISCLAIMER, disclosure_hits=hits, quantitative=checks,
                method_priorities=[
                    dict(category=code, label=label, question=question,
                         review_priority="P1" if code in ("RESERVE_MODEL", "CREDIT_MIGRATION") else
                         "P2" if code in ("MODIFICATIONS", "NONCASH_INTEREST") else "P3",
                         mentions=sum(h["category"] == code for h in hits))
                    for code, label, pattern, question in PATTERNS],
                documents_scanned=sum(d.get("status") in ("OK", "TEXT_EXTRACTED") for d in documents),
                review_flags=sum(c["status"] == "REVIEW_FLAG" for r in checks for c in r["checks"]),
                not_testable=sum(c["status"] == "NOT_TESTABLE" for r in checks for c in r["checks"]),
                concealment_proven=False, ai_specific_loss_proven=False,
                warning="A 0 találat/0 flag nem jelenti azt, hogy a banknál nincs rejtett probléma.")
