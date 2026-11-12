"""Aligned, wrapped Unicode console tables with optional ASCII fallback."""
from __future__ import annotations

import unicodedata
import re
import textwrap

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def width(text):
    return sum(0 if unicodedata.combining(c) or c in ("\ufe0f", "\u200d") else
               2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in ANSI.sub("", str(text)))


def pad(text, size, right=False):
    spaces = " "*max(0, size-width(text))
    return spaces+str(text) if right else str(text)+spaces


def wrap(text, size):
    """Wrap by display cells, including long URLs, not Python character count."""
    lines, current = [], ""
    for token in str(text).replace("\r", " ").replace("\n", " ").split():
        if current and width(current+" "+token) <= size:
            current += " "+token
            continue
        if current:
            lines.append(current)
            current = ""
        for char in token:
            if width(current+char) > size:
                lines.append(current)
                current = ""
            current += char
    if current or not lines:
        lines.append(current)
    return lines


def table(headers, rows, sizes, numeric=(), ascii_only=False):
    if len(headers) != len(sizes) or any(s < 4 for s in sizes):
        raise ValueError("Érvényes táblázat-oszlopszélességek kellenek.")
    h, v = ("-", "|") if ascii_only else ("─", "│")
    borders = ("+", "+", "+", "+", "+", "+", "+", "+", "+") if ascii_only else (
        "┌", "┬", "┐", "├", "┼", "┤", "└", "┴", "┘")
    def border(a, b, c):
        return a+b.join(h*(s+2) for s in sizes)+c
    def render(row):
        if len(row) != len(sizes):
            raise ValueError("Eltérő számú táblaoszlop.")
        cells = [wrap(c, s) for c, s in zip(row, sizes)]
        return [v+" "+(" "+v+" ").join(pad(cell[i] if i < len(cell) else "", sizes[j],
                                            j in numeric) for j, cell in enumerate(cells))+" "+v
                for i in range(max(map(len, cells)))]
    lines = [border(*borders[:3]), *render(headers), border(*borders[3:6])]
    for row in rows:
        lines += render(row)
    lines.append(border(*borders[6:]))
    return lines


def render(result, ascii_only=False, details=False):
    from macro_ai_credit_monitor import fmt
    icons = {"ok": "[OK]", "no": "[X]", "missing": "[?]", "warn": "[!]", "up": "^", "down": "v", "same": "=",
             "new": "[NEW]"} if ascii_only else {
                 "ok": "✅", "no": "❌", "missing": "⚪", "warn": "⚠", "up": "↑", "down": "↓", "same": "→", "new": "🆕"}
    def direction(value, unit="%"):
        from macro_ai_credit_monitor import number
        n = number(value)
        return "N/A" if n is None else f"{icons['up'] if n > 0 else icons['down'] if n < 0 else icons['same']} {fmt(n)}{unit}"
    def tbl(headers, rows, sizes, numeric=()):
        return table(headers, rows, sizes, numeric, ascii_only)
    def section(title):
        return ["", title, ("-" if ascii_only else "─")*100]
    r = result
    lines = ["="*100, f"AI / BANKHITEL / INFLÁCIÓ ELEMZŐ {r['version']} | PAPER / READ-ONLY",
             "KUTATÁSI / PAPER RENDSZER; NEM CRASH-JÓSLAT", "="*100,
             f"Értékelés: {r['asof']}",
             f"{icons['warn']} Earnings-snapshot: {r['cache_status']} | kor: {r['cache_age_hours']:.1f} óra",
             "Jelmagyarázat: ✅ adat/szabály ellenőrizve; ❌ nem igazolt; ⚠ vizsgálandó; ↑↓ változásirány."
             if not ascii_only else "Jelmagyarázat: [OK] ellenőrizve; [X] nem igazolt; [!] vizsgálandó; ^v irány.",
             "A pipa NEM banki egészségigazolás. A piros X NEM bizonyított veszteség."]
    if r.get("replay"):
        lines += [icons["warn"]+" MENTETT FUTÁS ÚJRAÉRTÉKELÉSE: NINCS MOSTANI HÁLÓZATI LEKÉRÉS.",
                  "Az alábbi forrásadatok és időpontok a régi futásból származnak."]
    refresh = r.get("public_refresh")
    lines += section("FRISS LEKÉRÉS ÉS AZ ELŐZŐ FUTÁSHOZ KÉPESTI VÁLTOZÁSOK")
    changes = r.get("run_changes", {})
    if changes.get("first_baseline"):
        lines.append(f"{icons['warn']} Első baseline: az elsőként látott adat nem feltétlenül ma publikált adat.")
    events = changes.get("changes", [])
    lines += tbl(["Ellenőrzés", "Eredmény"], [
        ["Lekérés", r.get("refresh_mode", "SNAPSHOT_ONLY")],
        ["Új / változott megfigyelések", str(len(events))],
        ["Korábban megőrzött adatkulcsok", str(changes.get("remembered_total", 0))],
        ["Automatikus order", icons["no"]+" NINCS"]], [34, 62])
    for event in events[:10 if not details else len(events)]:
        lines.append(f"{icons['new']} {event['kind']}: {event['key']}")
    if len(events) > 10 and not details:
        lines.append("További változások a JSON-ban / --details módban.")
    if changes.get("warning"):
        lines.append(icons["warn"]+" "+changes["warning"])
    if refresh:
        lines += tbl(["Bank", "Új SEC forrás keresése"], [
            [ticker, status] for ticker, status in refresh.get("bank_fetch_status", {}).items()], [7, 89])
        lines += [icons["warn"]+" "+message for message in refresh.get("messages", [])]
    else:
        lines.append(icons["warn"]+" Offline snapshot; nincs mostani hálózati ellenőrzés.")
    lines += section("BANKI EARNINGS: DÁTUMOZOTT PROVIDER-SNAPSHOT, NEM AZ ÚJ SEC SZÖVEG EPS-E")
    lines += tbl(["Bank", "Időszak", "EPS tény", "EPS várt", "EPS eltérés", "Revenue eltérés"],
                 [[b["ticker"], b["period"] or "N/A", fmt(b["actual_eps"]), fmt(b["consensus_eps"]),
                   direction(b["eps_surprise_pct"]), direction(b["revenue_surprise_pct"])]
                  for b in r["earnings"]], [6, 10, 10, 10, 17, 20], (2, 3, 4, 5))
    lines.append("Az új SEC earnings-közlés nem írja felül csendben ezt a régi tény/konszenzus párt.")
    lines += section("ÁLTALÁNOS HITELMUTATÓK: SNAPSHOT / NEM AI-SZEGMENS")
    latest_credit = {}
    for row in r["credit_facts"]:
        if row["ticker"] not in latest_credit or row["period_end"] > latest_credit[row["ticker"]]["period_end"]:
            latest_credit[row["ticker"]] = row
    def billions(value):
        from macro_ai_credit_monitor import number
        v = number(value)
        return "N/A" if v is None else f"{v/1e9:.3f}"
    lines += tbl(["Bank", "Időszak", "Prov. Mrd USD", "QoQ %", "Gross Mrd USD", "ACL Mrd USD", "ACL/gross %"],
                 [[c["ticker"], c["period"], billions(c.get("provision_usd")), direction(c.get("provision_qoq_pct")),
                   billions(c.get("gross_loans_usd")), billions(c.get("allowance_usd")),
                   fmt(c.get("allowance_to_gross_pct"))] for c in latest_credit.values()],
                 [6, 9, 13, 14, 14, 12, 13], (2, 3, 4, 5, 6))
    lines.append("Publication-proxy / definíció-egyezés nem igazolt; provision-emelkedés nem AI-default.")
    if refresh:
        lines += section("FRISS SEC JELENTÉSEK / EARNINGS-ESEMÉNYEK")
        lines += tbl(["Bank", "Form", "Befogadási idő", "Accession", "Típus"],
                     [[f["ticker"], f["form"], f["accepted_at"], f["accession"], f["status"]]
                      for f in refresh.get("filings", [])], [6, 7, 25, 21, 26])
        lines += section("FRISS BLS CPI / PPI: INDEXBŐL SZÁMOLT VÁLTOZÁS")
        lines += tbl(["Sorozat", "Időszak", "Index", "Változás", "Időalap", "Eltérés pp", "Surprise státusz"],
                     [[p["series"], p["period"], f"{p['value']:.3f}", direction(p["change_pct"]),
                       p["basis"], direction(p.get("surprise", {}).get("difference"), " pp"),
                       p.get("surprise", {}).get("status", "CONSENSUS_NA")]
                      for p in refresh.get("prices", [])], [26, 7, 9, 13, 8, 12, 23], (2, 3, 5))
        lines.append("A friss PPI referenciahónapja változhat, a naptár önmagában nem actual.")
        lines += tbl(["Bank", "SEC filing időszak", "Provision Mrd USD", "ACL Mrd USD", "Státusz"],
                     [[b["ticker"], b.get("period", "N/A"), billions(b.get("provision_usd")),
                       billions(b.get("allowance_usd")), b["status"]]
                      for b in refresh.get("banks", [])], [6, 17, 18, 14, 35], (2, 3))
    lines += section("KÉSLELTETETT MEGJELENÍTÉS: ADVERSARIAL HIPOTÉZIS, NEM ELTITKOLÁSI VÁD")
    audit = r.get("forensics", {})
    lines += [icons["warn"]+" "+audit.get("disclaimer", "Nem vizsgált."),
              f"Szövegesen vizsgált dokumentumok: {audit.get('documents_scanned', 0)} | "
              f"Összes tárolt negyedév: review={audit.get('review_flags', 0)} | "
              f"nem értékelhető={audit.get('not_testable', 0)}",
              icons["no"]+" Bizonyított eltitkolás / AI-specifikus veszteség: NINCS IGAZOLVA."]
    lines += tbl(["Prioritás", "Vizsgált mechanizmus", "Szöveges említések"], [
        [p["review_priority"], p["label"], str(p["mentions"])]
        for p in sorted(audit.get("method_priorities", []), key=lambda p: p["review_priority"])],
        [10, 58, 23])
    lines.append("P1/P2/P3 = ellenőrzési sorrend, NEM eltitkolási valószínűség. Említés != rossz hír.")
    lines += ["", "RÖVIDÍTÉSEK ÉS AZ ÖT ELLENŐRZÉS JELENTÉSE",
              *tbl(["Jelölés", "Közérthetően / mit keres a szabály?"], [
                  ["NPL / nonaccrual", "Problémás / nem kamatoztatott hitelállomány; nem minden NPL-definíció azonos. Itt a támogatott nonaccrual-adat szerepel."],
                  ["ACL", "Várható hitelveszteségekre elszámolt tartalékállomány; nem készpénzkassza."],
                  ["Provision / prov.", "Az időszak hitelveszteségi ráfordítása/visszaírása; nem azonos az ACL állományával."],
                  ["NCO", "Nettó hitelleírás = veszteségként leírt hitelek mínusz korábbi leírásokból visszanyert pénz."],
                  ["PIK", "Payment-in-kind: nem készpénzes kamat. Növekedése ellenőrzési kérdés, nem automatikus fizetésképtelenség."],
                  ["Hitel/tart.", "Két változás együtt: nonaccrual-hitel nő ÉS tartalék csökken. A / NEM hányados."],
                  ["NCO/prov", "Nettó leírás nő ÉS új veszteségi ráfordítás csökken. A / NEM hányados."],
                  ["Módosítás", "Nőtt-e az azonos definíciójú módosított hitelállomány?"],
                  ["ACL egyezés", "Záró tartalék = nyitó tartalék + provision - NCO + összes egyéb változás."],
                  ["NINCS JEL", "Csak a konkrét szabály nem teljesült; nem általános banki egészségigazolás."],
              ], [20, 90])]
    if audit.get("automatic_extraction"):
        lines.append("Automatikus hiteladat-normalizálás: AKTÍV (XBRL + ellenőrzött earnings táblasémák).")
        lines.append("Eredmény: macro_credit_normalized.json; ismeretlen táblaszerkezetből nem tippelünk számot.")
        last_available = {}
        for row in r.get("normalized_credit_facts", []):
            if row.get("available_at") and any(row.get(k) is not None for k in
                    ("npl_usd", "allowance_usd", "provision_usd", "net_chargeoffs_usd")):
                last_available[row["ticker"]] = row
        if last_available:
            lines += ["", "LEGUTÓBBI GÉPILEG KINYERT ADATOK (Mrd USD): NEM FELTÉTLENÜL AKTUÁLIS EARNINGS",
                      *tbl(["Bank", "Időszak", "Nonaccrual", "Loan ACL", "Loan prov.", "NCO"],
                           [[row["ticker"], row["period_end"], *[billions(row.get(k)) for k in
                             ("npl_usd", "allowance_usd", "provision_usd", "net_chargeoffs_usd")]]
                            for row in last_available.values()], [6, 10, 14, 14, 14, 14], (2, 3, 4, 5))]
            scoped = [row for row in last_available.values() if row.get("scope_description")]
            if scoped:
                lines += ["", "MIT TARTALMAZNAK A BANKI SZÁMOK? NEM AZONOS BANKKÖZI NPL-DEFINÍCIÓK",
                          *tbl(["Bank", "Elfogadott portfólió / definíció"],
                               [[row["ticker"], row["scope_description"]] for row in scoped], [6, 98])]
                all_rows = {(row["ticker"], row["period_end"]): row
                            for row in r.get("normalized_credit_facts", [])}
                changes = []
                for row in scoped:
                    before = [old for (bank, period), old in all_rows.items()
                              if bank == row["ticker"] and period < row["period_end"]]
                    previous = max(before, key=lambda old: old["period_end"]) if before else None
                    if not previous or previous.get("scope_id") != row.get("scope_id"):
                        continue
                    def change(field):
                        a, b = previous.get(field), row.get(field)
                        old = previous.get("metric_provenance", {}).get(field)
                        new = row.get("metric_provenance", {}).get(field)
                        if (a is None or b is None or not old or not new or
                                old.get("definition_id") != new.get("definition_id")):
                            return "ADATHIÁNY"
                        arrow = icons["up"] if b>a else icons["down"] if b<a else "="
                        return f"{a/1e9:.3f} {arrow} {b/1e9:.3f}"
                    changes.append([row["ticker"], previous["period_end"]+" -> "+row["period_end"],
                                    *[change(k) for k in ("npl_usd", "allowance_usd",
                                                         "net_chargeoffs_usd", "provision_usd")]])
                if changes:
                    lines += ["", "KÖZVETLEN NEGYEDÉVES VÁLTOZÁSOK (Mrd USD; nyíl = irány, nem minősítés)",
                              *tbl(["Bank", "Időszakpár", "Nonaccrual", "Hitel ACL", "NCO", "Loan prov."],
                                   changes, [6, 24, 19, 19, 19, 19])]
        unhandled = audit.get("extraction_diagnostics", [])
        if unhandled:
            from macro_credit_autodata import compact_diagnostics
            compact = compact_diagnostics(unhandled)
            occurrences = sum(x["occurrences"] for x in compact)
            lines.append(f"{icons['warn']} Kinyerési diagnosztika: {occurrences} előfordulás / {len(compact)} egyedi bank-tag-ok-forrás csoport; NEM banki riasztások.")
        coverage = r.get("credit_coverage", {})
        if coverage:
            lines += ["", f"ADATLEFEDETTSÉG: {coverage['window_start']} .. {coverage['window_end']} (8 egymást követő negyedév)",
                      *tbl(["Bank", "Célidőszak", "Kitöltött mező", "Utolsó elfogadott időszak", "Kizárt régi fact"],
                           [[b["ticker"], b["target_period"], str(len(b["current_fields"])),
                             b["latest_accepted_period"] or "Nincs az ablakban",
                             str(b["excluded_old_fact_count"])] for b in coverage["banks"]],
                           [6, 10, 14, 25, 15], (2, 4))]
            lines.append(f"Új szabály szerint visszaolvasható, megőrzött jelölt-fact: {coverage.get('restored_candidate_fact_count', 0)} (nem ennyi új probléma).")
            lines.append(f"Earnings HTML-ből felismert forrásmegfigyelés: {coverage.get('reviewed_earnings_fact_count', 0)} (ismétlést is tartalmazhat).")
            if coverage["banks"] and all(not b["current_fields"] and b.get("target_candidate_facts") == 0
                                          for b in coverage["banks"]):
                lines.append(icons["warn"]+" A célnegyedévhez nincs megtartott címkézett szám sem az elfogadott, sem a jelölt bemenetben.")
                lines.append("Ez nem a számok közzétételének hiánya: az earnings HTML-táblákhoz külön adapter/forrásvizsgálat kellhet.")
            if not coverage.get("source_values_retained"):
                lines.append(icons["warn"]+" A régi exportban az elutasított értékek/kontextusok nem maradtak meg; új lekérés kell, nem következtetünk az adat nem publikálására.")
    latest = {}
    for row in audit.get("quantitative", []):
        latest[row["ticker"]] = row
    def status(c):
        return {"REVIEW_FLAG": icons["warn"]+" REVIEW",
                "NO_TRIGGER": icons["ok"]+" NINCS JEL",
                "NOT_TESTABLE": icons["missing"]+" ADATHIÁNY"}[c]
    lines += tbl(["Bank", "Időszak", "Hitel/tart.", "NCO/prov", "Módosítás", "PIK", "ACL egyezés"],
                 [[row["ticker"], row["period_end"], *[status(c["status"]) for c in row["checks"]]]
                  for row in latest.values()], [6, 10, 13, 13, 13, 13, 13])
    lines.append("ADATHIÁNY = nem értékelhető. Nem negatív teszt, nem banki bukás és nem bizonyított veszteség.")
    lines.append("Az időszak az utolsó lezárt naptári negyedév is lehet; ez NEM jelenti, hogy már közzétették.")
    current_checks = [c for row in latest.values() for c in row["checks"]]
    lines.append(f"Célnegyedév: {sum(c['status']=='REVIEW_FLAG' for c in current_checks)} review / "
                 f"{sum(c['status']=='NO_TRIGGER' for c in current_checks)} lefutott, jel nélkül / "
                 f"{sum(c['status']=='NOT_TESTABLE' for c in current_checks)} nem értékelhető.")
    lines.append("A fenti összesített számláló történeti negyedéveket is tartalmaz; nem csak mai riasztások.")
    problems = []
    for row in latest.values():
        for check in row["checks"]:
            if check["status"] == "NOT_TESTABLE":
                problems.append([row["ticker"], row["period_end"], check["label"],
                                 " ".join(check.get("missing_reasons") or [check["explanation"]])])
    if problems:
        replacements = {
            "npl_usd": "nonaccrual/problémás hitel",
            "allowance_usd": "hitelveszteségi tartalék",
            "net_chargeoffs_usd": "nettó hitelleírás",
            "provision_usd": "hitelveszteségi ráfordítás",
            "modified_loans_usd": "módosított hitelállomány",
            "pik_interest_usd": "nem készpénzes kamat",
            "acl_other_changes_usd": "összes egyéb tartalékváltozás",
            "[NO_SUPPORTED_REPORTED_CONCEPT]": "[a kinyerő nem fogadott el támogatott adatot]",
            "[NO_ACCEPTED_INPUT_FACT_NOT_PROOF_OF_NONDISCLOSURE]": "[nincs elfogadott gépi bemenet; nem bizonyítja a közlés hiányát]",
            "[QUARTER_FLOW_NOT_AVAILABLE]": "[nincs elfogadható negyedéves mozgásadat]",
            "[NO_COMPARABLE_MODIFIED_LOAN_STOCK_IN_REVIEWED_EARNINGS]": "[nincs azonos definíciójú módosított hitelállomány a támogatott earnings táblákban]",
            "[NO_SEPARATELY_IDENTIFIED_QUARTER_PIK_IN_REVIEWED_EARNINGS]": "[külön negyedéves PIK-kamat nem azonosítható a támogatott earnings sémákból]",
            "[FUNDED_ONLY_QUARTER_PROVISION_NOT_IDENTIFIED]": "[a csak folyósított hitelek negyedéves provisionja nem azonosított; a tágabb credit provision nem helyettesíti]",
            "[FUNDED_ONLY_OTHER_CHANGES_NOT_IDENTIFIED_NOT_ASSUMED_ZERO]": "[a csak folyósított hitelek összes egyéb ACL-változása nem azonosított; nem tekintjük nullának]",
            "[FUNDED_NONACCRUAL_STOCK_NOT_IDENTIFIED]": "[nem azonosított a megfelelő nonaccrual-állomány]",
            "[QUARTER_NET_NOT_GROSS_CHARGEOFFS_NOT_IDENTIFIED]": "[nincs megfelelő negyedéves nettó leírás; a bruttó charge-off nem helyettesíti]",
            "[FIELD_NOT_IDENTIFIED_IN_REVIEWED_EARNINGS_LAYOUT]": "[a támogatott earnings táblákban nem azonosított mező]",
        }
        for problem in problems:
            for old, new in replacements.items():
                problem[-1] = problem[-1].replace(old, new)
        lines += ["", "HIÁNYZÓ ADATOK / DEFINÍCIÓK BANKONKÉNT",
                  *tbl(["Bank", "Időszak", "Ellenőrzés", "Mi hiányzik / miért nem futott?"],
                       problems, [6, 10, 22, 68])]
    usable = []
    for row in audit.get("quantitative", []):
        if row not in latest.values() and any(c["status"] != "NOT_TESTABLE" for c in row["checks"]):
            usable.append(row)
    if usable:
        previous_latest = {}
        for row in usable:
            previous_latest[row["ticker"]] = row
        lines += ["", "KORÁBBI, ÉRTÉKELHETŐ NEGYEDÉVEK: NEM A LEGFRISSEBB EARNINGS",
                  *tbl(["Bank", "Időszak", "NPL/ACL", "NCO/prov", "Módosítás", "PIK", "ACL mérleg"],
                       [[r["ticker"], r["period_end"], *[status(c["status"]) for c in r["checks"]]]
                        for r in previous_latest.values()], [6, 10, 13, 13, 13, 13, 13])]
    hits = audit.get("disclosure_hits", [])
    for i, hit in enumerate(hits[:12 if not details else len(hits)], 1):
        lines += [f"{icons['warn']} R{i:02d} {hit['ticker']} | {hit['label']} | "
                  f"AI szó ugyanebben a kontextusban: {'IGEN, nem oksági igazolás' if hit['ai_context_present'] else 'NEM'}",
                  *["  "+v for v in wrap(hit["excerpt"], 96)],
                  *["  Ellenőrizendő: "+v for v in wrap(hit["question"], 78)],
                  "  Forrás: "+(hit.get("source") or "N/A")]
    if len(hits) > 12 and not details:
        lines.append("További szövegtalálatok a JSON-ban / --details módban.")
    lines.append(audit.get("warning", "Nincs lefedettségi igazolás."))
    lines += section("GDP ÉRZÉKENYSÉG: MINDEN BEMENET FELTEVÉS")
    s = r["scenario_assumptions"]
    lines.append(f"Nominális={s['nominal_gdp_pct']}% | alapdeflátor={s['baseline_deflator_pct']}% | {s['basis']}")
    lines += tbl(["Alt. defl. %", "AI pp", "Alap reál %", "AI nélkül alap %", "Alt. reál %", "AI nélkül alt. %"],
                 [[f"{g['alternative_deflator_pct']:.1f}", f"{g['ai_contribution_pp']:.1f}",
                   fmt(g["reported_real_pct"]), fmt(g["no_ai_baseline_pct"]),
                   fmt(g["alternative_real_pct"]), fmt(g["no_ai_alternative_pct"])]
                  for g in r["gdp_scenarios"]], [12, 6, 13, 17, 13, 17], range(6))
    lines += section("INDEX / GOLD: NEM KALIBRÁLT FELTÉTELES STRESSZ, NEM FAIR VALUE")
    lines += tbl(["Index", "Feltétel", "EPS %", "Multiple %", "Árhatás %", "Szint"],
                 [[e["asset"], e["name"], fmt(e["eps_change_pct"]), fmt(e["multiple_change_pct"]),
                   direction(e["change_pct"]), fmt(e["scenario_level"])]
                  for e in r["equity_scenarios"]], [7, 20, 10, 12, 18, 12], (2, 3, 4, 5))
    lines += tbl(["Gold feltétel", "Reálhozam pp", "USD %", "Árhatás %"],
                 [[g["name"], fmt(g["real_yield_change_pp"]), fmt(g["dollar_change_pct"]),
                   direction(g["change_pct"])] for g in r["gold_scenarios"]], [39, 16, 15, 20], (1, 2, 3))
    c = r["credit_scenario"]
    lines += section("AI-HITEL VÁRHATÓ VESZTESÉG: EAD / PD / LGD MIND FELTEVÉS")
    lines += tbl(["EAD Mrd USD", "Alap PD/LGD %", "Stressz PD/LGD %", "Alap EL Mrd", "Stressz EL Mrd", "Delta Mrd"],
                 [[billions(c["exposure_usd"]), f"{c['baseline_pd_pct']} / {c['baseline_lgd_pct']}",
                   f"{c['stress_pd_pct']} / {c['stress_lgd_pct']}",
                   billions(c["baseline_loss_usd"]), billions(c["stress_loss_usd"]),
                   billions(c["incremental_loss_usd"])]], [13, 15, 17, 13, 15, 12], (0, 3, 4, 5))
    lines += section("NEM IGAZOLT KÖVETKEZTETÉSEK / FORRÁSÜTKÖZÉSEK")
    lines += [icons["no"]+" AI okozta teljes infláció: nem azonosított.",
              icons["no"]+" Hétfői crash időzítése és valószínűsége: nem számítható ebből.",
              icons["no"]+" Igazolt AI nélküli GDP / indokolt piaci esés: N/A."]
    lines.append("A következő forrásütközések a provider-snapshotra vonatkoznak, nem a friss BLS blokkra.")
    for note in r["source_conflicts"]+r["notes"]:
        lines += [icons["warn"]+" "+v for v in wrap(note, 96)]
    if details:
        lines += section("RÉSZLETES MAKRÓ SNAPSHOT / NEM FRISS BLS BLOKK")
        lines += tbl(["Mutató", "Érték", "Referencia", "Surprise"], [
            [p["name"], str(p["value"])+" "+p["unit"], p["period_end"], p["surprise"]["status"]]
            for p in r["macro"]], [39, 19, 12, 29])
    lines += ["", "Végrehajtási engedély: NEM. Kutatás/paper; nincs order vagy pozíciókezelés.", "="*100]
    return lines
