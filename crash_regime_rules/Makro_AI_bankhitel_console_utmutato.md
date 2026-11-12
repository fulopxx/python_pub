# AI, bankhitel és infláció: konzolos paper stresszelemző

A 0.3.3 verzió elkülöníti a lekért, dátumozott adatokat, az ellenőrizetlen
oksági hipotéziseket és a szabadon paraméterezhető stresszforgatókönyveket.
Nem épít be „hétfői crash” következtetést. A korábbi GC/NQ/VIX/QQQ
belépési tiltásokhoz nincs bekötve, és semmilyen ordert nem küld.

## Telepítés

Az alábbi fájlokat tedd a meglévő `crash_regime_rules` mappába:

| Fájl | Szerep |
|---|---|
| `macro_ai_credit_monitor.py` | Konzolriport, számítások és JSON/TXT export |
| `macro_public_data.py` | Opcionális SEC/BLS nyilvános HTTP-adapter |
| `macro_credit_forensics.py` | Adversarial szövegvizsgálat és negyedéves ellenőrzések |
| `macro_credit_autodata.py` | Automatikus SEC companyfacts / inline-XBRL negyedéves hiteladat-normalizálás |
| `macro_credit_tables.py` | Bankonként ellenőrzött earnings HTML-tábla- és GS szöveges közlésadapterek |
| `macro_monitor_state.py` | Előző megfigyelések és érték/revízióváltozások megőrzése |
| `macro_console.py` | Igazított Unicode táblák, ikonok és ASCII fallback |
| `macro_consensus.json` | Alapból üres, opcionális CPI/PPI konszenzusbemenet |
| `macro_provider_snapshot.json` | A most lekért, dátumozott finance-adatok portable snapshotja |
| `macro_scenarios.json` | Átírható stresszfeltevések; nem piaci tényadat |
| `build_macro_provider_snapshot.py` | Új finance-CSV exportokból snapshot építése, offline |
| `macro_release_calendar_2026_10.json` | Külön ellenőrzött, dátumhoz kötött release-naptár |
| `test_macro_ai_credit_monitor.py` | Offline, hálózat nélkül futó regressziós tesztek |
| `test_macro_monitor_v020.py` | Újrafuttatás, forensic szabályok, filing-keresés, táblázatok tesztjei |
| `test_macro_credit_autodata.py` | Automatikus kinyerés, időszak/scope/deviza és adathiány tesztjei |
| `test_macro_credit_v031.py` | Újabb címkék, valós diagnosztikai alakok, időablak, forráscache és replay tesztjei |
| `replay_macro_credit_report.py` | Korábbi JSON újraértékelése hálózat és élő állapotmódosítás nélkül |
| `package_macro_earnings_sources.py` | A helyileg már meglévő nyilvános earnings HTML-ek hash-ellenőrzött ZIP-be gyűjtése |
| `test_macro_credit_v032.py` | Megőrzött jelöltek, WFC scope, kerekítések és forráscsomagolás tesztjei |
| `test_macro_credit_v033.py` | Valódi, rövidített earnings forrásminták és a HTML-/ZIP-kinyerés regressziói |
| `tests/fixtures/earnings_033/` | A feltöltött nyilvános források rövidített, stílus nélküli táblái és eredeti forrásjegyzéke |

Python 3.10+ szükséges, ezen a rétegen nincs új pip-függőség.
Az alapfutás TWS és új előfizetés nélkül is működik:

```powershell
python macro_ai_credit_monitor.py
```

**0.2.0-tól az alapfutás minden alkalommal új SEC/BLS lekérést próbál indítani.**
A SEC-hez az alábbi User-Agent szükséges; nélküle a banki frissítés kimarad,
nem lesz csendben frissnek nevezve a snapshot. Csak offline megjelenítéshez:

```powershell
python macro_ai_credit_monitor.py --offline
```

A kimenet `macro_ai_output/macro_ai_latest.json` és
`macro_ai_output/macro_ai_console.txt`. PyCharmban a scriptet közvetlenül
is futtathatod. Minden szcenárió-beállítást a `macro_scenarios.json` fájlban
változtass: ez az alapból betöltött, ténylegesen használt konfiguráció.

## Mit tud, és mit nem?

| Réteg | Most működik | Korlát |
|---|---|---|
| Banki earnings | JPM, BAC, C, WFC, GS, MS; EPS tény/várt és revenue meglepetés | A csatolt fájl dated snapshot; a consensus nem frissül SEC-ből |
| Általános hiteladat | Provision, QoQ változás, gross loans, allowance és azonos időszaki arány | Nem AI-szegmensadat; több banknál taghiány van |
| Automatikus hiteladat | Standard XBRL és hat bankhoz ellenőrzött earnings sémák; negyedévpárok, proveniencia | Nem általános táblázatértelmező; új sémánál adathiány maradhat |
| AI-specifikus veszteség | Külön jelzi az igazolatlanságot | A gépi hiteladat nem automatikusan AI-szegmensadat; CET1/AI-EAD nincs normalizálva |
| CPI/PPI | Referencia-időszak, mért érték, konszenzus hiányának jelzése | Az előző érték nem konszenzus; az aktuális release-meglepetés nincs igazolva |
| Makró release-naptár | Külön ellenőrzött dátumok | Nem actual, nem meglepetés; nem automatikusan frissülő naptár |
| GDP AI nélkül | AI pp × alternatív GDP-deflátor érzékenységi rács | Nem tényszerű AI-hozzájárulás-becslés |
| NQ/DJI | EPS és értékeltségi multiple együttes stressze | Nem GDP-ből származó fair value; a két index ugyanazt a szemléltető rácsot kapja |
| Gold | Reálhozam/USD/menedék/likvidáció log-faktor stressz | Szabadon választott, nem becsült együtthatók; nem előrejelzés |
| AI-hitel stressz | EAD × PD × LGD, alap és stressz várható veszteség | Az EAD/PD/LGD kizárólag feltevés, nem a hat bank mért aggregátuma |
| Piaci árfolyam | Saját referencia megadható a configban | Élő NQ/DJI/GC jegyzést nem kér le; hiányában nincs kitalált target |
| Végrehajtás | Kizárólag konzol és fájlexport | Nincs order, számlalekérés vagy pozíciókezelés |

Az EPS/revenue meglepetés önmagában nem banki fizetőképességi stressz.
A provision növekedésének oka lehet portfóliónövekedés, várakozás,
módszertan és más hitelszegmens is. Az allowance állomány, a provision
időszaki ráfordítás; nem adhatók össze veszteségként.

## Helyi friss lekérés és ismételt futtatás

A SEC companyfacts API kulcs nélkül elérhető, és a benyújtott XBRL tényadatokat
szolgáltatja; nem azonos a legújabb earnings sajtóközleménnyel vagy a call-lal.
Az endpointokat a [SEC API dokumentációja](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
rögzíti. A helyi adapterhez valós kapcsolattartási User-Agentet adj meg, ne API-kulcsot.

```powershell
$env:SEC_USER_AGENT = "SajatResearch valos-kapcsolati-email-cimed"
python macro_ai_credit_monitor.py --refresh-public
```

A példában a teljes értéket cseréld le saját kutatási azonosítóra és valódi,
`@` jelet tartalmazó kapcsolattartási címre. A program nem írja a User-Agentet
a riportba. PyCharmban a Run Configuration environment variables mezőjében
is beállítható. Enélkül a SEC-blokk kimarad, a BLS-kérés továbbra is megkísérelhető.

A BLS kérés négy fix CPI/PPI indexsort kér le. Az indexekből azonos sorozat
egyéves/havi változása számolódik; a szezonálisan igazított és nem igazított
sorok elkülönülnek. Az alap POST-sémát a
[BLS API dokumentációja](https://www.bls.gov/developers/api_signature_v2.htm)
adja meg. A kulcs nélküli kvóta és a hálózati hozzáférés miatt részleges válasz
vagy 403 is előfordulhat; az elemző ilyenkor hibát/hiányt mutat, nem „OK”-t.

**A nyilvános adapter implementált és mock-válaszokkal tesztelt; a 0.3.0
kinyerést itt nem ellenőriztük éles SEC/BLS HTTP-adatokon.** A felhasználó
korábbi 0.2.0 futásában a SEC/BLS kapcsolat működött. Ez nem igazolja
automatikusan az új kinyerő mezőnkénti éles lefedettségét.

SEC-korlátok:

- **Filing-lag:** a friss earnings már ismert lehet, amikor a companyfacts még csak
  a korábbi 10-Q adatát tartalmazza. Az új 2.02 earnings 8-K és legutóbbi
  10-Q/10-K már kereshető és szövegesen olvasható, de az earnings táblák
  EPS/consensus oszlopai nincsenek normalizáltan kinyerve.
- **Időszak:** csak közvetlen, negyedév hosszúságú provision flow fogadható el;
  a YTD és éves flow nem lesz negyedévesként felcímkézve.
- **Taxonómia:** explicit standard us-gaap fogalomlista, companyfacts és
  inline-XBRL forrásból. Banki saját extension-tagok ellenőrizetlen jelöltek,
  nem kerülnek név alapján automatikusan jóváhagyott számszerű vizsgálatba.
- **Gross/net:** a net loans taget nem nevezzük bruttó hitelnek.
- **Revízió:** a jelenlegi XBRL vintage történeti auditnak nem alkalmas.
- **Konszenzus:** sem a SEC, sem a BLS nem ad piaci várakozást. A helyi frissítés
  ezért nem frissíti csendben a dated earnings- és consensus-snapshotot.

A `--asof` történeti ellenőrzéshez adható meg ISO időzónával. Nem azonos
point-in-time backtesttel. Az aktuális HTTP-frissítés történeti/jövőbeli
asof mellett tiltott, mert nem lehetne igazolni a megfelelő adat-vintage-ot.
ASOF-futás a `replay` almappába kerül, a live state-et nem módosítja.

Minden új release után ugyanazt a scriptet futtasd. Az első futás baseline,
nem „ma megjelent hírek” listája. A további futások külön jelölik:

- **FIRST_SEEN_OBSERVATION:** új forráskulcs, időszak vagy accession.
- **VALUE_OR_REVISION_CHANGED:** ugyanazon megfigyelés értéke/filing-metaadata változott.
- **Nincs változás:** a most sikeresen ellenőrzött kulcsok között nem lett eltérés;
  nem jelenti azt, hogy minden lehetséges adatforrást lefedtünk.
- **Hiba / hiány:** a sikertelen forrás nem törli a korábbi jó megfigyelést.
  A live riport a hibát mutatja; a régi adat csak explicit offline cache-ként használható.

Új fájlok az output mappában: `macro_ai_state.json`, `macro_public_cache.json`,
valamint legfeljebb harminc futási JSON a `history` mappában.
Azonos lekérési időbélyeg helyett a tényleges megfigyelésértékek változását figyeljük.
A state nyilvános adatokat tárol, nem TWS-számla- vagy pozícióinformációkat.

Az archívumkeresés bankonként egy legutóbbi 2.02 earnings eseményre és két
külön időszakú legutóbbi 10-Q/K-ra korlátozott, 240 napos ablakban. Nem teljes EDGAR-archívum.
Az earnings mellékletek közül legfeljebb hat, releváns nevű HTML-fájlnév
jelöltet olvasunk; ez nem igazolt EX-99 típusbesorolás. A forrás minősítése
ezt jelzi, a kimaradt PDF vagy eltérő mellékletnév vakfolt marad.

## Adversarial hitelminőségi vizsgálat

A modell feltételezett késleltetési lehetőségeket keres, nem feltételezi,
hogy egy konkrét bank ténylegesen eltitkol valamit. A P1/P2/P3 sorrend
elemzői ellenőrzési prioritás, nem mért valószínűség vagy banki csalásrangsor.

| Prioritás | Vizsgált mechanizmus | Kért ellenőrzés |
|---|---|---|
| P1 | Céltartalék-felszabadítás / CECL / modellváltozás | ACL roll-forward, veszteség és azonos portfólió |
| P1 | NPL/nonaccrual/késedelem/minősítésromlás | Értékek és definíciók változása a provisionhöz képest |
| P2 | Módosítás, forbearance, lejárat-hosszabbítás | Cash collection és utólagos teljesítés |
| P2 | PIK / kapitalizált kamat | Nem cash bevétel, kamatbeszedés és szerződés |
| P3 | Hitelértékesítés / átsorolás / securitization | Realizált veszteség, recourse és mixhatás |
| P3 | Mérlegen kívüli / VIE / unfunded exposure | Végső kockázatviselő és duplázásmentes kitettség |

A kulcsszó-találat **csak említés**, lehet általános, tagadó vagy kedvező közlés.
A program a szövegkörnyezetet és a forrás URL-jét adja meg, nem következtet
automatikusan negatív jelentésre. Az AI szó közeli jelenléte nem AI-veszteség-igazolás.
A 0 találat nem bizonyítja, hogy nincs probléma.

Az öt számszerű ellenőrzés:

- **NPL nő / ACL csökken.**
- **NCO nő / provision csökken.**
- **Módosított hitelállomány nő.**
- **PIK kamat nő.**
- **ACL roll-forward eltérés:** `ACL1 = ACL0 + provision - NCO + egyéb változások`.

Ezek csak azonos, explicit normalizáltnak jelölt portfólió és egymást követő
negyedévek esetén futhatnak. Az NCO/provision/PIK flow negyedéves, nem YTD.
Az „egyéb változás” nem feltételezett nulla. Hiányzó input esetén
**⚪ ADATHIÁNY** jelenik meg, külön táblában bankonként/ellenőrzésenként a
hiányzó negyedév, mező, definíció vagy forrás feltüntetésével.

### Automatikus bemenet 0.3.0-tól

Normál futásnál már NEM szükséges kézzel `--credit-details` fájlt készíteni.
Az automatikus folyamat a következő:

- **Lekérés:** companyfacts és a már letöltött jelentések inline-XBRL címkéi.
  A vizuálisan szomszédos, de nem címkézett táblázatszámokból nem talál ki adatot.
- **Hatókör:** a hat bank, USD, dimenzió nélküli vállalati kontextus. Az inline
  kontextus CIK-jének egyeznie kell a figyelt bank CIK-jével.
- **Időszak:** pontosan az utolsó nyolc egymást követő naptári negyedév.
  A hiányzó frissebb időszakok helyét nem tölti ki 2021–2022-es adattal.
  Az utolsó lezárt negyedév megjelenítése nem állítja, hogy már publikálták.
- **Áramlásadat:** közvetlen negyedév előnyben. YTD-különbség csak azonos
  filing/accession, közzétételi vintage, koncepció, naptári évkezdet mellett.
  A „csak éves/YTD adat van” önmagában nem elegendő.
- **NCO:** közvetlen támogatott net write-off fogalom vagy ugyanazon filing
  azonos negyedévének leírása mínusz visszanyerése, negatív komponens nélkül.
- **Fogalmak:** a nonaccrual fogalom nem minden bank teljes NPL/NPA állománya.
  A tágabb „összes hitelveszteség provision” nem lesz automatikusan
  azonos a szűkebb loan allowance portfólió provisionjével.
- **Proveniencia:** mezőnként tag, összeg, időszak, forrás, accession,
  megismerhetőség és számítási módszer. Definícióváltáskor az összevetés megáll.
- **Konfliktus:** ugyanazon vintage eltérő számai nem oldódnak fel találomra.
- **Export:** `macro_ai_output/macro_credit_normalized.json`; a fő riportban
  is szerepel `normalized_credit_facts` néven.
- **Diagnosztika:** `macro_credit_diagnostics.json` bankonkénti lefedettséggel;
  előfordulások és egyedi bank/címke/ok/forrás csoportok külön.
- **Nyers forrás:** a `sec_sources` alkönyvtárban tartalomhash-es companyfacts
  JSON és letöltött HTML. Ezek a helyi nyilvános forrásfájlok nagyok lehetnek,
  a mappa növekedhet. A fő riport az elutasított releváns jelöltek értékét és
  kontextusát is őrzi; ezek nem válnak ettől jóváhagyott banki mutatóvá.
- **Korábbi időszak:** ha a friss negyedév hiányos, külön megjelenik a korábbi
  értékelhető negyedév, nem aktuális earningsként átcímkézve.
- **Újrafuttatás:** az automatikus számszerű hiteladatok változása is része
  az előző futáshoz viszonyított megfigyeléskövetésnek.

**A folyamat automatikus, a publikált adatok lefedettsége nem garantált.**
A PIK, a módosított hitelek és az összes egyéb ACL-átvezetés nem feltétlenül
van külön, támogatott gépi címkével közzétéve. Ezekből nem gyártunk
látszólag teljes JSON-t: az output megmondja, mi maradt hiányos.
Nem támogatott saját címkéhez ellenőrzött bank-specifikus mapping szükséges.
Az eltérő, régi provision fogalom nem választhatja ki automatikusan a 2019-es
negyedévet a bank legfrissebb negyedéve helyett.

### A 0.3.3 kiegészítése

Az automatikus friss SEC-HTML feldolgozásba bekerültek a bankonként
ellenőrzött earnings táblák. Ez nem csak megjelenítési javítás: a kinyert
értékek az automatikusan előállított normalizált JSON-ba és az öt banki
ellenőrzésbe is bekerülnek. Kézi `--credit-details` fájl nem szükséges.

- **Ellenőrzés:** banki SEC CIK/URL/accession, befogadási idő, explicit
  oszlopdátum, milliós USD-egység, sorcím és oszlopgeometria.
- **Proveniencia:** eredeti forrás, teljes HTML SHA256, tábla/sor,
  dátumoszlopok, nyers sor, scope és adapterverzió.
- **JPM:** csak megtartott nonaccrual-hitelek, consumer + wholesale
  összege. A HFS/FV hitel nem kerül az ugyanitt használt loan ACL mellé.
- **Citi:** a növekvő dátumsorrendet kezeli; loan ACL, loan provision,
  nettó hitelveszteség és jelentett egyéb mozgás. Nem az ACLUC összegét használja.
- **BAC/WFC:** a tágabb credit-loss provision nem lesz csak folyósított
  hitelek provisionjaként felhasználva. Ez néhány ellenőrzést jogosan blokkol.
- **GS:** két pontosan azonosított szöveges közlés, funded allowance és
  negyedéves NCO. Nem általános narratív pénzügyiadat-kinyerés.
- **MS:** a dátumozott loan ACL-mátrixból loan ACL és negyedéves loan
  provision. A bruttó charge-off nem helyettesíti a nettó NCO-t.
- **Összehasonlíthatóság:** külön earnings profilok, támogatott
  összehasonlító negyedévekkel. Régi XBRL-adatot nem told hozzá csendben.
  Scope-váltás új baseline a változáskövetésben, nem pénzügyi romlás.
- **PIK/módosítás:** e források támogatott sémáiból nem azonosítható teljes,
  összehasonlítható idősor; nincs nullával feltöltés.

A feltöltött 15 teljes HTML újrafeldolgozásakor a 42 célnegyedéves mezőből
20 kitölthető, a 30 ellenőrzésből 8 értékelhető. A 8 lefutott szabály
nem jelzett triggerfeltételt; a többi 22 nem értékelhető.
Ez nem egészségigazolás és nem AI-hitelveszteség bizonyítása vagy kizárása.

Korábbi export + forráscsomag hálózat nélkül:

```powershell
python replay_macro_credit_report.py macro_ai_output/macro_ai_latest.json --earnings-zip macro_ai_output/macro_earnings_sources.zip --output-dir macro_credit_replay_033
```

A replay az eredeti export `asof` idejét megtartja, és nem módosítja az élő
állapotot. Friss napi adathoz a normál `macro_ai_credit_monitor.py` futás kell.
Csak a régi cache `--offline` megnyitása nem tölti visszamenőleg az új HTML-adatokat.

### A 0.3.2 kiegészítése

- **Megőrzött jelöltek:** pontosan támogatott standard fogalom, SEC URL,
  banki CIK, USD és dimenzió nélküli kontextus alapján visszaolvashatók.
  Nincs címkenév-hasonlóság alapján végzett automatikus besorolás.
- **Új flow-fogalmak:** a ténylegesen megfigyelt standard negyedéves
  credit-loss expense és nettó write-off fogalmak is szerepelnek.
- **WFC külön scope:** explicit taglista a hitelek és nettó lízingbefektetések
  együtt kezelt portfóliójára; nem keveredik a csak hitel portfólióval.
- **Nettó leírás ellenőrzése:** ahol ugyanazon filingban a bruttó leírás és
  a visszanyerés is szerepel, a jelentett nettó értéket egyezteti.
- **Kerekítés:** eltérő pontosságú, de egymással összeegyeztethető közlések
  nem lesznek hamis konfliktusok; a pontosabb értéket választja.
- **Régi export korlátja:** az inline jelöltek namespace URI-ja a régi
  JSON-ban nem szerepel. A provenance ezt kifejezetten jelöli; az engedélyezés
  a pontos qualified name, hivatalos SEC-forrás és CIK-kontextus ellenőrzésére épül.

A 0.3.1-es exportból több Q2 adat állítható helyre, de a Q3 számok ott
nincsenek meg sem elfogadott, sem jelölt XBRL-bemenetként. Ez nem bizonyítja,
hogy a bank nem közölte a számokat. A 0.3.3 a később feltöltött
nyers earnings HTML-csomag támogatott tábláit már fel tudja dolgozni.

Ezek csomagolásához nem szükséges új API-futás:

```powershell
python package_macro_earnings_sources.py
```

A `macro_ai_output/macro_earnings_sources.zip` csak a helyi, hash szerint
egyező nyilvános earnings HTML-eket és forrásjegyzéket tartalmazza.
Konfigurációs JSON, SEC-kapcsolattartó vagy számlaadat nem kerül a csomagba.

### A 0.3.0-ból származó régi export korlátja

A felhasználó éles 0.3.0-s exportjában sok elutasított címkéhez csak név/ok
maradt meg, érték és XBRL-kontextus nem. Ezeket egy új taglista nem tudja
utólag visszaállítani. A 0.3.1-hez új normál futás szükséges; a kézi JSON
megadása továbbra sem kötelező. A támogatott modern nonaccrual- és
allowance-fogalmak listája bővült, de a nem címkézett earnings-táblákra
ez nem általános pénzügyi táblázatkinyerő.

Korábbi riport újraértékelhető külön könyvtárba, HTTP nélkül:

```powershell
python replay_macro_credit_report.py macro_ai_output/macro_ai_latest.json --output-dir macro_credit_replay
```

Ha a forrásfájlokat már az új verzió mentette, a `--source-dir
macro_ai_output/sec_sources` opcióval az explicit megadott könyvtárból
újraolvashatók. A report metaadatában tárolt tartalomhashnek egyeznie kell.
Nem követ tetszőleges, JSON-ban kapott helyi fájlútvonalat.

Opcionális, ellenőrzött kézi felülbírálás továbbra is átadható JSON-listából:

```powershell
python macro_ai_credit_monitor.py --credit-details sajat_normalizalt_hiteladatok.json
```

Egy sor szükséges kulcsai: `ticker`, `period_end`, `available_at` időzónával,
`scope_id`, `normalization_verified: true`, `flow_basis: "QUARTER"` és `source`.
A mért mezők: `npl_usd`, `allowance_usd`, `provision_usd`, `net_chargeoffs_usd`,
`modified_loans_usd`, `pik_interest_usd`, `acl_other_changes_usd`.
Két összehasonlítható negyedévi sor kell. A normalizálási jelző a felhasználó/
provider állítása, nem a program által végzett audit.

## CPI/PPI konszenzus és a szebb konzol

A `macro_consensus.json` alapból üres. Soronként megadható a `series_id`,
`period` (YYYY-MM), `basis` (MOM_SA vagy YOY_NSA), `unit: "percent"`, `value`,
`source`, `recorded_at` és `scheduled_release_at` időzónás mezőkkel.
Csak a release előtt rögzített, azonos időalapú és időszakú pár különbsége
számolható. A BLS-válasz nem igazolja a tényleges release időpontját, ezért
az eredmény leíró összevetés, nem teljesen igazolt surprise-vintage.

Az új táblák Unicode megjelenítési szélességgel igazítanak és tördelnek.
✅ adat/szabály ellenőrizve, ⚪ ADATHIÁNY = nem értékelhető,
❌ nem igazolt oksági állítás, ⚠ kézi review,
↑↓ változásirány. A pipa nem banki egészségigazolás.

```powershell
python macro_ai_credit_monitor.py --ascii
python macro_ai_credit_monitor.py --details
python macro_ai_credit_monitor.py --offline --details
```

Az `--ascii` az ikonokat/kereteket cseréli, a magyar ékezetek megmaradnak.
Monospaced font szükséges; egyes terminálok eltérően jeleníthetik meg az emojikat.
A `--legacy-console` csak a régi szöveges nézet, az új auditok a JSON-ban megmaradnak.

## A GDP-számítás

A BEA szerint az implicit deflátor a nominális és a megfelelő chained-dollar
érték hányadosából származik; a
[BEA definíciója](https://www.bea.gov/help/faq/513)
nem jogosít fel a CPI automatikus GDP-deflátorként történő használatára.
Azonos időszakú nominális és deflátor-változás esetén:

\[
g_{\mathrm{real}}=\frac{1+g_{\mathrm{nominal}}}{1+\pi_{\mathrm{GDP}}}-1.
\]

A már reál GDP-ből nem vonunk le még egyszer CPI-t. Az ANNUAL, YOY és QOQ_SAAR
csak saját, azonos időalapú feltevésekkel használható; az éves CPI nem keverhető
egyszerű negyedéves növekedéssel.

Az alapkonfiguráció teljesen szemléltető: 6% nominális növekedés,
3.7% alap GDP-deflátor és 3.2 pp AI-hozzájárulás. Nem az aktuális
gazdaság paramétereinek becslése, és nem az AI nélküli −1% „bizonyítása”.

| Szemléltető eredmény | Érték |
|---|---:|
| Alap reálnövekedés | +2.22% |
| AI nélkül, alapdeflátorral | −0.98% |
| 10% alternatív GDP-deflátorral, AI levonása nélkül | −3.64% |
| Ugyanaz az AI nélküli nominális ellenforgatókönyv, 10% deflátorral | −6.65% |

Az AI nélküli alternatív sorban ugyanazt a nominális ellenforgatókönyvet
árazzuk át. Nem vonunk le még egyszer változatlan AI pp-t az alternatív
reálnövekedésből. A chained-volume hozzájárulások egyszerű kivonása csak
érzékenységi közelítés: import, kiszorítás, multiplikátorok, bázis és
komponens-árak nélkül nem oksági nemzetiszámla-modell.

**A 10%-os deflátor önmagában nem jelent −10%-os GDP-növekedést.**
Azonos időalapon −10% reálnövekedéshez és +10% deflátorhoz −1% nominális
növekedés tartozna. Ez matematikai feltétel, nem az aktuális GDP állítása.

## Részvényindex- és gold-szcenáriók

A részvényindex stresszének képlete:

\[
\frac{P_1}{P_0}
=(1+\Delta EPS)(1+\Delta \mathrm{multiple}).
\]

Például −10% EPS és −20% multiple együtt −28% feltételes árhatás,
nem −30%. Ez nem a GDP-ből következő „indokolt esés”, és nem hétfői target.
A modell nem kalibrálja külön a NQ és a DJI eredményérzékenységét,
indexsúlyait vagy forward értékeltségét. Mindkettő ugyanazt az illusztratív
rácsot kapja; valós instrumentumspecifikus modellezés későbbi fejlesztés.

A gold-blokk képlete:

\[
r_{\log}=\beta_{RY}\Delta RY_{\mathrm{pp}}
+\beta_{USD}\Delta USD_{\mathrm{fraction}}
+s_{\mathrm{haven}}+s_{\mathrm{liquidity}},
\qquad r=\exp(r_{\log})-1.
\]

Az alap együtthatók nem regresszióból származnak. A menedék/likvidáció
mezők loghozam-százalékpontként szerepelnek, nem közvetlen százalékos
árhatásként. A negatív és pozitív példa sem előrejelzés; az infláció
önmagában nem dönti el az arany irányát.

## Naptár és a kiinduló tézis

A CPI szeptemberi közlését október 14. 08:30 ET-re, a PPI-ét október 15.
08:30 ET-re sorolja a [BLS októberi naptára](https://www.bls.gov/schedule/2026/10_sched.htm).
A naptár nem actual és nem consensus, tehát nem igazolja a „magasabb lett
a vártnál” állítást. A csatolt naptár dátumhoz kötött, később kézzel frissítendő.

A kész rendszer nem fogja igazoltként kiírni, hogy kizárólag az AI okozta
az inflációt, vagy hogy biztos hétfői crash következik. Ezekhez sem oksági
azonosítás, sem megbízható crash-időzítő modell nincs a bemenetekben.
A friss makró actualhoz konszenzus esetén annak forrását, referencia-időszakát,
unitját és release előtti vintage-át is külön ellenőrizni kell.

## Új finance-export importálása

A `build_macro_provider_snapshot.py` már letöltött finance-CSV fájlokat olvas,
nem hitelesíti és nem hívja a Computer connectorát a Windows gépedről.
Új earnings/consensus frissítéshez új provider-export kell.

```powershell
python build_macro_provider_snapshot.py --finance-dir "C:\finance_exports" --calendar macro_release_calendar_2026_10.json
python macro_ai_credit_monitor.py
```

Az exportmappában csak egy-egy vintage legyen a várt fájlnév-mintákból.
Többszörös találatnál a builder leáll. Hiányzó fájlt nem tölt ki nullával.
Az inkompatibilis időszaki mérleget nem keveri a frissebb eredményadathoz.
Az input snapshot forrásfájljait a konzol és JSON listázza.

## Ellenőrzés

A makrómodul külön offline tesztjei lefuttathatók az alábbi paranccsal.
A tesztek között teljes automatikus bemenetképzés és CLI-export is van,
szintetikus, nem tényleges banki adatokkal. A SEC/BLS tényleges hálózati
hozzáférés és a bankonkénti éles mezőlefedettség nem része ennek az eredménynek.
A 0.3.3 új regressziói valódi, rövidített forrásmintákon is ellenőrzik
a kinyerést, nem csak szintetikus factokon. A feltöltött teljes forráscsomag
hash-ellenőrzött, hálózatmentes újrafeldolgozása külön ellenőrzés;
új éles SEC-lekérést ez nem jelent.

```powershell
python -m unittest test_macro_ai_credit_monitor -v
python -m unittest test_macro_monitor_v020 -v
python -m unittest test_macro_credit_autodata -v
python -m unittest test_macro_credit_v031 -v
python -m unittest test_macro_credit_v032 -v
python -m unittest test_macro_credit_v033 -v
```

A tesztek a deflálást, annualizálási segédképleteket, AI pp kezelését,
nemlineáris EPS/multiple hatást, gold kétirányú érzékenységét, EAD/PD/LGD
domént, jövőbeli/stale adatok megkülönböztetését és a SEC/BLS parser
mock-válaszait ellenőrzik. Nem bizonyítják a gazdasági hipotézisek
igazságát vagy a szcenáriók valószínűségét.

Ez kutatás és elemzés, nem személyre szabott pénzügyi tanács.
Befektetési döntés előtt kérj megfelelően képzett pénzügyi tanácsadótól segítséget.
