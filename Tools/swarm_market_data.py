"""
Swarm Intelligence Trading — Market Data Layer
===============================================
yfinance data fetching, feature engineering, correlation computation, and caching.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from .persistent_cache import PersistentTTLCache, market_cache_settings


# ── Constants ──────────────────────────────────────────────────────────────────

# NSE sector indices (yfinance symbols). Used as sector benchmarks.
SECTOR_ETFS = {
    'IT':          '^CNXIT',
    'Banking':     '^NSEBANK',
    'Pharma':      '^CNXPHARMA',
    'Auto':        '^CNXAUTO',
    'FMCG':        '^CNXFMCG',
    'Metal':       '^CNXMETAL',
    'Energy':      '^CNXENERGY',
    'Realty':      '^CNXREALTY',
    'Infra':       '^CNXINFRA',
    'PSU_Bank':    '^CNXPSUBANK',
    'Media':       '^CNXMEDIA',
}

# Broad-market ETFs on NSE used as market leaders (have proper OHLCV + volume)
MARKET_LEADERS = ['NIFTYBEES.NS', 'BANKBEES.NS', 'ITBEES.NS', 'JUNIORBEES.NS']

# Liquid NSE large-caps per sector for sector-first scanning
SECTOR_STOCKS: Dict[str, List[str]] = {
    "Basic_Materials": [
      "ULTRACEMCO.NS",
      "JSWSTEEL.NS",
      "VEDL.NS",
      "TATASTEEL.NS",
      "HINDZINC.NS",
      "ASIANPAINT.NS",
      "HINDALCO.NS",
      "GRASIM.NS",
      "PIDILITIND.NS",
      "SOLARINDS.NS",
      "JINDALSTEL.NS",
      "AMBUJACEM.NS",
      "SHREECEM.NS",
      "LLOYDSME.NS",
      "NATIONALUM.NS",
      "NMDC.NS",
      "SAIL.NS",
      "JSL.NS",
      "LINDEINDIA.NS",
      "COROMANDEL.NS",
      "FACT.NS",
      "APLAPOLLO.NS",
      "UPL.NS",
      "BERGEPAINT.NS",
      "HINDCOPPER.NS",
      "PIIND.NS",
      "JKCEMENT.NS",
      "FLUOROCHEM.NS",
      "DALBHARAT.NS",
      "NAVINFLUOR.NS",
      "WELCORP.NS",
      "ACC.NS",
      "HSCL.NS",
      "RAMCOCEM.NS",
      "SHYAMMETL.NS",
      "BAYERCROP.NS",
      "KIOCL.NS",
      "SUMICHEM.NS",
      "SARDAEN.NS",
      "DEEPAKNTR.NS",
      "ACUTAAS.NS",
      "GPIL.NS",
      "ATUL.NS",
      "CHAMBLFERT.NS",
      "TATACHEM.NS",
      "INDIACEM.NS",
      "JSWCEMENT.NS",
      "GALLANTT.NS",
      "RATNAMANI.NS",
      "CENTURYPLY.NS",
      "AETHER.NS",
      "BASF.NS",
      "AARTIIND.NS",
      "KANSAINER.NS",
      "EIDPARRY.NS",
      "IGIL.NS",
      "JAINREC.NS",
      "ANURAS.NS",
      "FINEORG.NS",
      "SPLPETRO.NS",
      "AKZOINDIA.NS",
      "DEEPAKFERT.NS",
      "USHAMART.NS",
      "JINDALSAW.NS",
      "VINATIORGA.NS",
      "PARADEEP.NS",
      "NSLNISP.NS",
      "PRIVISCL.NS",
      "NUVOCO.NS",
      "PCBL.NS",
      "JUBLINGREA.NS",
      "SANDUMA.NS",
      "SHARDACROP.NS",
      "SHAILY.NS",
      "LLOYDSENT.NS",
      "STARCEMENT.NS",
      "GRWRHITECH.NS",
      "JAYNECOIND.NS",
      "MAHSEAMLES.NS",
      "JKLAKSHMI.NS",
      "CLEAN.NS",
      "ACI.NS",
      "IMFA.NS",
      "BIRLACORPN.NS",
      "ALKYLAMINE.NS",
      "RCF.NS",
      "GSFC.NS",
      "JKPAPER.NS",
      "GNFC.NS",
      "GALAXYSURF.NS",
      "SUDARSCHEM.NS",
      "MIDHANI.NS",
      "PRSMJOHNSN.NS",
      "INDIAGLYCO.NS",
      "MOIL.NS",
      "KINGFA.NS",
      "JAIBALAJI.NS",
      "ASHAPURMIN.NS",
      "RALLIS.NS",
      "EPIGRAL.NS",
      "SURYAROSNI.NS",
      "MIDWESTLTD.NS",
      "GUJALKALI.NS",
      "KSCL.NS",
      "MBAPL.NS",
      "DHANUKA.NS",
      "SUNFLAG.NS",
      "GHCL.NS",
      "BANSALWIRE.NS",
      "GULFOILLUB.NS",
      "RAIN.NS",
      "RESPONIND.NS",
      "INDIGOPNTS.NS",
      "NACLIND.NS",
      "STYRENIX.NS",
      "CHEMPLASTS.NS",
      "NEOGEN.NS",
      "GOODLUCK.NS",
      "NFL.NS",
      "POCL.NS",
      "LXCHEM.NS",
      "KRISHANA.NS",
      "VISHNU.NS",
      "MANINDS.NS",
      "BALAMINES.NS",
      "HEIDELBERG.NS",
      "FOSECOIND.NS",
      "SAMBHV.NS",
      "ELLEN.NS",
      "ADVENZYMES.NS",
      "RPEL.NS",
      "WSTCSTPAPR.NS",
      "STEELCAS.NS",
      "KSL.NS",
      "ORIENTCEM.NS",
      "TATVA.NS",
      "NOCIL.NS",
      "BHAGCHEM.NS",
      "MAITHANALL.NS",
      "POLYPLEX.NS",
      "FCL.NS",
      "GREENPLY.NS",
      "BHARATRAS.NS",
      "VENUSPIPES.NS",
      "GREENPANEL.NS",
      "JUBLCPL.NS",
      "JTLIND.NS",
      "XPROINDIA.NS",
      "PREMEXPLN.NS",
      "ROSSARI.NS",
      "EUROPRATIK.NS",
      "SIRCA.NS",
      "MANGLMCEM.NS",
      "VSSL.NS",
      "PRAKASH.NS",
      "KIRIINDUS.NS",
      "SAGCEM.NS",
      "SOTL.NS",
      "DDEVPLSTIK.NS",
      "VEEDOL.NS",
      "BEPL.NS",
      "TIRUMALCHM.NS",
      "ORISSAMINE.NS",
      "KCP.NS",
      "RAJRATAN.NS",
      "RHETAN.NS",
      "CAMLINFINE.NS",
      "INSECTICID.NS",
      "APCOTEXIND.NS",
      "MSPL.NS",
      "MUKANDLTD.NS",
      "SHK.NS",
      "JAICORPLTD.NS",
      "GSPCROP.NS",
      "YASHO.NS",
      "HITECH.NS",
      "IPL.NS",
      "SESHAPAPER.NS",
      "STALLION.NS",
      "DCW.NS",
      "VIDHIING.NS",
      "GODAVARIB.NS",
      "ARFIN.NS",
      "SANSTAR.NS",
      "MAHASTEEL.NS",
      "ASTEC.NS",
      "JGCHEM.NS",
      "GOCLCORP.NS",
      "SPIC.NS",
      "PUNJABCHEM.NS",
      "IGPL.NS",
      "ANDHRAPAP.NS",
      "SALASAR.NS",
      "IFGLEXPOR.NS",
      "MOL.NS",
      "PAUSHAKLTD.NS",
      "PLATIND.NS",
      "EXCELINDUS.NS",
      "TINNARUBR.NS",
      "BHARATWIRE.NS",
      "ANDHRSUGAR.NS",
      "STEELXIND.NS",
      "SHREEPUSHK.NS",
      "MADRASFERT.NS",
      "RATNAVEER.NS",
      "GULPOLY.NS",
      "BSHSL.NS",
      "SHREDIGCEM.NS",
      "RSL.NS",
      "AEROENTER.NS",
      "AVTNPL.NS",
      "MANALIPETC.NS",
      "TNPL.NS",
      "DHARMAJ.NS",
      "OAL.NS",
      "ZUARI.NS",
      "DECCANCE.NS",
      "SCODATUBES.NS",
      "ELECTHERM.NS",
      "HARIOMPIPE.NS",
      "ESTER.NS",
      "GEMAROMA.NS",
      "AMNPLST.NS",
      "BODALCHEM.NS",
      "RAMASTEEL.NS",
      "NCLIND.NS",
      "INDOBORAX.NS",
      "TNPETRO.NS",
      "PDMJEPAPER.NS",
      "HERANBA.NS",
      "NRAIL.NS",
      "MWL.NS",
      "SRHHYPOLTD.NS",
      "SUDARCOLOR.NS",
      "INDOAMIN.NS",
      "VALIANTORG.NS",
      "MAANALU.NS",
      "UNIENTER.NS",
      "KUANTUM.NS",
      "KOTHARIPET.NS",
      "SALSTEEL.NS",
      "BHAGERIA.NS",
      "SAURASHCEM.NS",
      "ADVANCE.NS",
      "FAIRCHEMOR.NS",
      "DMCC.NS",
      "GANESHBE.NS",
      "SATIA.NS",
      "BESTAGRO.NS",
      "KAMDHENU.NS",
      "MMP.NS",
      "AGARIND.NS",
      "BHAGYANGR.NS",
      "JAYAGROGN.NS",
      "PREMIERPOL.NS",
      "20MICRONS.NS",
      "CHEMFAB.NS",
      "CHEMCON.NS",
      "PRIMO.NS",
      "SUKHJITS.NS",
      "KHAICHEM.NS",
      "SADHNANIQ.NS",
      "ACL.NS",
      "BMWVENTLTD.NS",
      "ARIES.NS",
      "DICIND.NS",
      "ORIENTCER.NS",
      "KRONOX.NS",
      "SURAJLTD.NS",
      "RAMAPHO.NS",
      "SHIVAUM.NS",
      "OCCLLTD.NS",
      "LORDSCHLO.NS",
      "PAKKA.NS",
      "EMAMIPAP.NS",
      "SHAH.NS",
      "SHIVALIK.NS",
      "VRAJ.NS",
      "SHALPAINTS.NS",
      "VISASTEEL.NS",
      "VINYLINDIA.NS",
      "CHEMBONDCH.NS",
      "IGCL.NS",
      "ORIENTPPR.NS",
      "PLASTIBLEN.NS",
      "KOTYARK.NS",
      "MANORG.NS",
      "RUBFILA.NS",
      "MANAKSTEEL.NS",
      "BEDMUTHA.NS",
      "APCL.NS",
      "AARTISURF.NS",
      "RUCHIRA.NS",
      "NATHBIOGEN.NS",
      "HPAL.NS",
      "GENUSPAPER.NS",
      "KANORICHEM.NS",
      "KESORAMIND.NS",
      "GOACARBON.NS",
      "DYNPRO.NS",
      "SRD.NS",
      "ASAHISONG.NS",
      "DIAMINESQ.NS",
      "PODDARMENT.NS",
      "GSLSU.NS",
      "NOVAAGRI.NS",
      "NAGAFERT.NS",
      "VSTL.NS",
      "STARPAPER.NS",
      "VMSTMT.NS",
      "SHREYANIND.NS",
      "INDOUS.NS",
      "SIKKO.NS",
      "CHEMBOND.NS",
      "VIKASECO.NS",
      "DSFCL.NS",
      "RUDRA.NS",
      "VASWANI.NS",
      "ARVEE.NS",
      "MANAKALUCO.NS",
      "KAMOPAINTS.NS",
      "TAINWALCHM.NS",
      "MCL.NS",
      "CENTEXT.NS",
      "INCREDIBLE.NS",
      "ISHANCH.NS",
      "KRITIKA.NS",
      "ARCHIDPLY.NS",
      "VITAL.NS",
      "SAMPANN.NS",
      "MAGNUM.NS",
      "AKSHARCHEM.NS",
      "IVP.NS",
      "HPIL.NS",
      "NARMADA.NS",
      "SHAHALLOYS.NS",
      "WIPL.NS",
      "CUBEXTUB.NS",
      "HINDCON.NS",
      "OSWALSEEDS.NS",
      "RAJMET.NS",
      "JOCIL.NS",
      "GOYALALUM.NS",
      "NEUEON.NS",
      "SHYAMCENT.NS",
      "BVCL.NS",
      "ZENITHSTL.NS",
      "NAVKARURB.NS",
      "KAKATCEM.NS",
      "HISARMETAL.NS",
      "PRAKASHSTL.NS",
      "BONLON.NS",
      "ALKALI.NS",
      "SUNDARAM.NS",
      "AGROPHOS.NS",
      "VINEETLAB.NS",
      "MALUPAPER.NS",
      "SANGINITA.NS",
      "ABMINTLLTD.NS",
      "ASHOKAMET.NS",
      "SAGARDEEP.NS",
      "GFSTEELS.NS",
      "SEYAIND.NS",
      "SUPREMEENG.NS",
      "KRIDHANINF.NS",
      "SHRENIK.NS",
      "ANKITMETAL.NS",
      "ASTRON.NS",
      "BILVYAPAR.NS",
      "BOHRAIND.NS",
      "IMPEXFERRO.NS",
      "ARENTERP.NS",
      "OMKARCHEM.NS"
    ],
    "Communication_Services": [
      "BHARTIARTL.NS",
      "INDUSTOWER.NS",
      "IDEA.NS",
      "BHARTIHEXA.NS",
      "NAUKRI.NS",
      "TATACOMM.NS",
      "PFOCUS.NS",
      "SUNTV.NS",
      "AFFLE.NS",
      "INDIAMART.NS",
      "NAZARA.NS",
      "PVRINOX.NS",
      "RAILTEL.NS",
      "TTML.NS",
      "ZEEL.NS",
      "TIPSMUSIC.NS",
      "SAREGAMA.NS",
      "NETWORK18.NS",
      "JUSTDIAL.NS",
      "DBCORP.NS",
      "MPSLTD.NS",
      "ROUTE.NS",
      "NAVNETEDUL.NS",
      "IMAGICAA.NS",
      "HATHWAY.NS",
      "BCG.NS",
      "MTNL.NS",
      "JAGRAN.NS",
      "DEN.NS",
      "SIGNPOST.NS",
      "BALAJITELE.NS",
      "AQYLON.NS",
      "STLNETWORK.NS",
      "SUYOG.NS",
      "MATRIMONY.NS",
      "NDTV.NS",
      "GTPL.NS",
      "SANDESH.NS",
      "TVTODAY.NS",
      "DISHTV.NS",
      "SCHAND.NS",
      "HTMEDIA.NS",
      "ENIL.NS",
      "HMVL.NS",
      "ONMOBILE.NS",
      "ZEEMEDIA.NS",
      "RKSWAMY.NS",
      "NDLVENTURE.NS",
      "CINELINE.NS",
      "SHEMAROO.NS",
      "RCOM.NS",
      "UFO.NS",
      "RADIOCITY.NS",
      "DGCONTENT.NS",
      "TIPSFILMS.NS",
      "SAMBHAAV.NS",
      "RAJTV.NS",
      "BTML.NS",
      "GTL.NS",
      "MUKTAARTS.NS",
      "BAGFILMS.NS",
      "ODIGMA.NS",
      "CINEVISTA.NS",
      "TOUCHWOOD.NS",
      "SITINET.NS",
      "DNAMEDIA.NS",
      "PNC.NS",
      "CYBERMEDIA.NS",
      "NEXTMEDIA.NS",
      "TVVISION.NS",
      "RADAAN.NS",
      "SILLYMONKS.NS",
      "UNIINFO.NS",
      "CREATIVEYE.NS",
      "ORTEL.NS"
    ],
    "Consumer_Cyclical": [
      "MARUTI.NS",
      "TITAN.NS",
      "M&M.NS",
      "BAJAJ-AUTO.NS",
      "ETERNAL.NS",
      "EICHERMOT.NS",
      "TVSMOTOR.NS",
      "TMCV.NS",
      "HYUNDAI.NS",
      "TRENT.NS",
      "TMPV.NS",
      "MOTHERSON.NS",
      "BOSCHLTD.NS",
      "HEROMOTOCO.NS",
      "INDHOTEL.NS",
      "BHARATFORG.NS",
      "NYKAA.NS",
      "MEESHO.NS",
      "SWIGGY.NS",
      "UNOMINDA.NS",
      "SCHAEFFLER.NS",
      "MRF.NS",
      "VMM.NS",
      "KALYANKJIL.NS",
      "VOLTAS.NS",
      "BALKRISIND.NS",
      "IRCTC.NS",
      "PAGEIND.NS",
      "SONACOMS.NS",
      "ATHERENERG.NS",
      "ENDURANCE.NS",
      "ITCHOTELS.NS",
      "KPRMILL.NS",
      "FORCEMOT.NS",
      "JUBLFOOD.NS",
      "TVSHLTD.NS",
      "METROBRAND.NS",
      "EXIDEIND.NS",
      "APOLLOTYRE.NS",
      "ZFCVINDIA.NS",
      "MSUMI.NS",
      "AMBER.NS",
      "ASAHIINDIA.NS",
      "EIHOTEL.NS",
      "CIEINDIA.NS",
      "BELRISE.NS",
      "CRAFTSMAN.NS",
      "TRAVELFOOD.NS",
      "CHALET.NS",
      "SUNDRMFAST.NS",
      "OLAELEC.NS",
      "SHRIPISTON.NS",
      "VTL.NS",
      "CROMPTON.NS",
      "JBMA.NS",
      "SANSERA.NS",
      "CEATLTD.NS",
      "THELEELA.NS",
      "VENTIVE.NS",
      "GABRIEL.NS",
      "DEVYANI.NS",
      "THANGAMAYL.NS",
      "TRIDENT.NS",
      "LUMAXTECH.NS",
      "ABLBL.NS",
      "TBOTEK.NS",
      "JKTYRE.NS",
      "MINDACORP.NS",
      "FIRSTCRY.NS",
      "WELSPUNLIV.NS",
      "WHIRLPOOL.NS",
      "MANYAVAR.NS",
      "ARVIND.NS",
      "EUREKAFORB.NS",
      "BATAINDIA.NS",
      "CELLO.NS",
      "TIMETECHNO.NS",
      "LEMONTREE.NS",
      "CARTRADE.NS",
      "PNGJL.NS",
      "ASKAUTOLTD.NS",
      "BANCOINDIA.NS",
      "VARROC.NS",
      "PCJEWELLER.NS",
      "BLUESTONE.NS",
      "EPL.NS",
      "ABFRL.NS",
      "SAFARI.NS",
      "IXIGO.NS",
      "RELAXO.NS",
      "SEDEMAC.NS",
      "CAMPUS.NS",
      "WESTLIFE.NS",
      "V2RETAIL.NS",
      "PGIL.NS",
      "DYNAMATECH.NS",
      "PRICOLLTD.NS",
      "ALOKINDS.NS",
      "REDTAPE.NS",
      "AVL.NS",
      "ETHOSLTD.NS",
      "TTKPRESTIG.NS",
      "ARVINDFASN.NS",
      "GARFIBRES.NS",
      "SKYGOLD.NS",
      "SMLMAH.NS",
      "FIEMIND.NS",
      "SFL.NS",
      "SUPRAJIT.NS",
      "SJS.NS",
      "SAPPHIRE.NS",
      "GREENLAM.NS",
      "LGBBROSLTD.NS",
      "ITDC.NS",
      "SENCO.NS",
      "MHRIL.NS",
      "SYMPHONY.NS",
      "LUMAXIND.NS",
      "WAKEFIT.NS",
      "JAMNAAUTO.NS",
      "JUNIPER.NS",
      "VMART.NS",
      "SUBROS.NS",
      "RAYMONDLSL.NS",
      "GOKEX.NS",
      "THOMASCOOK.NS",
      "ICIL.NS",
      "SWARAJENG.NS",
      "SHARDAMOTR.NS",
      "RTNINDIA.NS",
      "VIPIND.NS",
      "BAJAJELEC.NS",
      "IFBIND.NS",
      "LUXIND.NS",
      "EMIL.NS",
      "GOLDIAM.NS",
      "STYLAMIND.NS",
      "JTEKTINDIA.NS",
      "RBA.NS",
      "BOSCH-HCIL.NS",
      "ORIENTELEC.NS",
      "VAIBHAVGBL.NS",
      "SANATHAN.NS",
      "SAMHI.NS",
      "AGI.NS",
      "WONDERLA.NS",
      "RAJESHEXPO.NS",
      "SSWL.NS",
      "KITEX.NS",
      "SHOPERSTOP.NS",
      "JINDALPOLY.NS",
      "TVSSRICHAK.NS",
      "KKCL.NS",
      "BOROLTD.NS",
      "SANDHAR.NS",
      "KDDL.NS",
      "GANECOS.NS",
      "EASEMYTRIP.NS",
      "CARRARO.NS",
      "UFLEX.NS",
      "BUILDPRO.NS",
      "AUTOAXLES.NS",
      "DPABHUSHAN.NS",
      "WHEELS.NS",
      "PARKHOTELS.NS",
      "NRBBEARING.NS",
      "FMGOETZE.NS",
      "MAYURUNIQ.NS",
      "JINDWORLD.NS",
      "CARYSIL.NS",
      "SIYSIL.NS",
      "BRIGHOTEL.NS",
      "TCPLPACK.NS",
      "EPACK.NS",
      "NITINSPIN.NS",
      "BOMDYEING.NS",
      "SANGAMIND.NS",
      "STYLEBAAZA.NS",
      "DIVGIITTS.NS",
      "TAJGVK.NS",
      "RML.NS",
      "SPAL.NS",
      "CANTABIL.NS",
      "FILATEX.NS",
      "LAOPALA.NS",
      "NILKAMAL.NS",
      "EIHAHOTELS.NS",
      "STOVEKRAFT.NS",
      "GNA.NS",
      "SHRINGARMS.NS",
      "STUDDS.NS",
      "MOLDTKPAC.NS",
      "LANDMARK.NS",
      "INDNIPPON.NS",
      "SPORTKING.NS",
      "HINDWAREAP.NS",
      "NDRAUTO.NS",
      "ORIENTHOT.NS",
      "COSMOFIRST.NS",
      "TALBROAUTO.NS",
      "RACLGEAR.NS",
      "YATRA.NS",
      "SUMEETINDS.NS",
      "PASHUPATI.NS",
      "RICOAUTO.NS",
      "DELTACORP.NS",
      "GOCOLORS.NS",
      "DOLLAR.NS",
      "RANEHOLDIN.NS",
      "KALAMANDIR.NS",
      "PRECAM.NS",
      "MOTISONS.NS",
      "ATULAUTO.NS",
      "SHANTIGOLD.NS",
      "HUHTAMAKI.NS",
      "ASIANHOTNR.NS",
      "IMPAL.NS",
      "KROSS.NS",
      "LGHL.NS",
      "RAJRILTD.NS",
      "WEL.NS",
      "HITECHGEAR.NS",
      "PNGSREVA.NS",
      "MIRCELECTR.NS",
      "RGL.NS",
      "UFBL.NS",
      "AYMSYNTEX.NS",
      "BUTTERFLY.NS",
      "HIMATSEIDE.NS",
      "MONTECARLO.NS",
      "BHARATSE.NS",
      "GRPLTD.NS",
      "RUPA.NS",
      "FAZE3Q.NS",
      "ORICONENT.NS",
      "LEMERITE.NS",
      "IGARASHI.NS",
      "BOROSCI.NS",
      "ROHLTD.NS",
      "BIL.NS",
      "JAYBARMARU.NS",
      "INDORAMA.NS",
      "TBZ.NS",
      "VHLTD.NS",
      "CENTENKA.NS",
      "ABCOTS.NS",
      "ARROWGREEN.NS",
      "ADVENTHTL.NS",
      "BORANA.NS",
      "NAHARSPING.NS",
      "MVGJL.NS",
      "GHCLTEXTIL.NS",
      "AMBIKCO.NS",
      "MUNJALAU.NS",
      "RADHIKAJWE.NS",
      "STANLEY.NS",
      "RUBYMILLS.NS",
      "ASAL.NS",
      "PVSL.NS",
      "SHREERAMA.NS",
      "RSWM.NS",
      "SARLAPOLY.NS",
      "PRECOT.NS",
      "MENONBE.NS",
      "NAHARPOLY.NS",
      "GLOSTERLTD.NS",
      "UNITEDPOLY.NS",
      "CHEVIOT.NS",
      "PATELRMART.NS",
      "COMSYN.NS",
      "HINDCOMPOS.NS",
      "IRISDOREME.NS",
      "SUTLEJTEX.NS",
      "PYRAMID.NS",
      "RBZJEWEL.NS",
      "COFFEEDAY.NS",
      "HLVLTD.NS",
      "TPLPLASTEH.NS",
      "MUFTI.NS",
      "KAMATHOTEL.NS",
      "SPECIALITY.NS",
      "ADVANIHOTR.NS",
      "MUNJALSHOW.NS",
      "RNBDENIMS.NS",
      "DONEAR.NS",
      "NAHARINDUS.NS",
      "FOCUS.NS",
      "KANPRPLA.NS",
      "LIBERTSHOE.NS",
      "MIRZAINT.NS",
      "RUSHIL.NS",
      "ORBTEXP.NS",
      "SREEL.NS",
      "TOLINS.NS",
      "PASUPTAC.NS",
      "SINCLAIR.NS",
      "BBTCL.NS",
      "MANOMAY.NS",
      "VARDMNPOLY.NS",
      "THOMASCOTT.NS",
      "BELLACASA.NS",
      "BANSWRAS.NS",
      "NDL.NS",
      "GINNIFILA.NS",
      "AXITA.NS",
      "REMSONSIND.NS",
      "RHL.NS",
      "SOMATEX.NS",
      "SETCO.NS",
      "VGL.NS",
      "BCONCEPTS.NS",
      "SHANKARA.NS",
      "AUTOIND.NS",
      "PPAP.NS",
      "SPENCERS.NS",
      "VARDHACRLC.NS",
      "PAVNAIND.NS",
      "BALAJEE.NS",
      "AHLEAST.NS",
      "SSDL.NS",
      "SUNDRMBRAK.NS",
      "SHIVAMAUTO.NS",
      "HITECHCORP.NS",
      "SINTERCOM.NS",
      "DCMNVL.NS",
      "OMAXAUTO.NS",
      "WORTHPERI.NS",
      "KALYANIFRG.NS",
      "TTL.NS",
      "CCHHL.NS",
      "UCAL.NS",
      "SONAMLTD.NS",
      "AHLWEST.NS",
      "BYKE.NS",
      "AERONEU.NS",
      "VIPCLOTHNG.NS",
      "ELGIRUBCO.NS",
      "SHIVATEX.NS",
      "SILGO.NS",
      "ZODIACLOTH.NS",
      "JMA.NS",
      "FILATFASH.NS",
      "KHADIM.NS",
      "SRTL.NS",
      "MARALOVER.NS",
      "BASML.NS",
      "URAVIDEF.NS",
      "MHLXMIRU.NS",
      "MODTHREAD.NS",
      "EMMBI.NS",
      "SUPERHOUSE.NS",
      "PILITA.NS",
      "BHARATGEAR.NS",
      "INDTERRAIN.NS",
      "AVROIND.NS",
      "SALONA.NS",
      "PRAXIS.NS",
      "FELDVR.NS",
      "LAL.NS",
      "ISFT.NS",
      "AIROLAM.NS",
      "WEIZMANIND.NS",
      "BSL.NS",
      "INDIANCARD.NS",
      "ZENITHEXPO.NS",
      "LAGNAM.NS",
      "RPPL.NS",
      "GLOBE.NS",
      "LAMBODHARA.NS",
      "SELMC.NS",
      "FIBERWEB.NS",
      "SURYALAXMI.NS",
      "LOVABLE.NS",
      "BHANDARI.NS",
      "LOYALTEX.NS",
      "SIL.NS",
      "MOKSH.NS",
      "DIGJAMLMTD.NS",
      "RELCHEMQ.NS",
      "SPLIL.NS",
      "AMDIND.NS",
      "DIGIDRIVE.NS",
      "PARASPETRO.NS",
      "ATLASCYCLE.NS",
      "PIONEEREMB.NS",
      "NAGREEKEXP.NS",
      "BANARBEADS.NS",
      "OSIAHYPER.NS",
      "KSR.NS",
      "TOKYOPLAST.NS",
      "PRITI.NS",
      "DELTAMAGNT.NS",
      "DAMODARIND.NS",
      "ORIENTLTD.NS",
      "VINNY.NS",
      "BALKRISHNA.NS",
      "AKI.NS",
      "JAIPURKURT.NS",
      "DANGEE.NS",
      "CELEBRITY.NS",
      "BANG.NS",
      "ARCHIES.NS",
      "HAVISHA.NS",
      "SHIVAMILLS.NS",
      "SHEKHAWATI.NS",
      "BLUECOAST.NS",
      "ADL.NS",
      "MITTAL.NS",
      "FEL.NS",
      "AKSHAR.NS",
      "SVPGLOB.NS",
      "KANANIIND.NS",
      "MOHITIND.NS",
      "SGL.NS",
      "VCL.NS",
      "EASTSILK.NS",
      "LIBAS.NS",
      "TGBHOTELS.NS",
      "SUPERSPIN.NS",
      "PEARLPOLY.NS",
      "FLFL.NS",
      "MORARJEE.NS",
      "LAXMICOT.NS",
      "MFML.NS",
      "FLEXITUFF.NS",
      "VIVIDHA.NS",
      "LYPSAGEMS.NS",
      "HEADSUP.NS",
      "KALYANI.NS",
      "ANTGRAPHIC.NS",
      "GLOBALE.NS",
      "WINSOME.NS",
      "NIRAJISPAT.NS",
      "RAJVIR.NS",
      "CLCIND.NS"
    ],
    "Consumer_Defensive": [
      "HINDUNILVR.NS",
      "ITC.NS",
      "DMART.NS",
      "NESTLEIND.NS",
      "VBL.NS",
      "BRITANNIA.NS",
      "GODREJCP.NS",
      "TATACONSUM.NS",
      "MARICO.NS",
      "UNITDSPR.NS",
      "DABUR.NS",
      "COLPAL.NS",
      "PATANJALI.NS",
      "UBL.NS",
      "RADICO.NS",
      "GODFRYPHLP.NS",
      "PGHH.NS",
      "PWL.NS",
      "GILLETTE.NS",
      "AWL.NS",
      "HATSUN.NS",
      "AVANTIFEED.NS",
      "EMAMILTD.NS",
      "ZYDUSWELL.NS",
      "BIKAJI.NS",
      "CCL.NS",
      "LTFOODS.NS",
      "ABDL.NS",
      "CUPID.NS",
      "GODREJAGRO.NS",
      "HONASA.NS",
      "TI.NS",
      "BBTC.NS",
      "BALRAMCHIN.NS",
      "TRIVENI.NS",
      "ORKLAINDIA.NS",
      "JYOTHYLAB.NS",
      "MANORAMA.NS",
      "KRBL.NS",
      "GAEL.NS",
      "DODLA.NS",
      "GOKULAGRO.NS",
      "RENUKA.NS",
      "BECTORFOOD.NS",
      "KWIL.NS",
      "PICCADIL.NS",
      "BAJAJCON.NS",
      "BANARISUG.NS",
      "NIITMTS.NS",
      "BAJAJHIND.NS",
      "VSTIND.NS",
      "CRIZAC.NS",
      "GRMOVER.NS",
      "GOPAL.NS",
      "VADILALIND.NS",
      "HERITGFOOD.NS",
      "DALMIASUG.NS",
      "GLOBUSSPR.NS",
      "PARAGMILK.NS",
      "SUNDROP.NS",
      "DIAMONDYD.NS",
      "GMBREW.NS",
      "VENKEYS.NS",
      "ADFFOODS.NS",
      "VINCOFE.NS",
      "TASTYBITE.NS",
      "ASALCBR.NS",
      "SDBL.NS",
      "VERANDA.NS",
      "ALLTIME.NS",
      "SULA.NS",
      "CLSEL.NS",
      "APEX.NS",
      "HMAAGRO.NS",
      "BCLIND.NS",
      "AVADHSUGAR.NS",
      "JARO.NS",
      "UTTAMSUGAR.NS",
      "SKMEGGPROD.NS",
      "DHAMPURSUG.NS",
      "NIITLTD.NS",
      "DWARKESH.NS",
      "KRISHIVAL.NS",
      "REGAAL.NS",
      "IFBAGRO.NS",
      "DBOL.NS",
      "GANESHCP.NS",
      "MAGADSUGAR.NS",
      "MUKKA.NS",
      "EIFFL.NS",
      "KNAGRI.NS",
      "DAVANGERE.NS",
      "GLOBAL.NS",
      "RAMANEWS.NS",
      "MODINATUR.NS",
      "APTECHT.NS",
      "MCLEODRUSS.NS",
      "SARVESHWAR.NS",
      "UGARSUGAR.NS",
      "FOODSIN.NS",
      "KAYA.NS",
      "MGEL.NS",
      "GOKUL.NS",
      "MAWANASUG.NS",
      "DCMSRIND.NS",
      "HARRMALAYA.NS",
      "KRITINUT.NS",
      "CPEDU.NS",
      "SCPL.NS",
      "COASTCORP.NS",
      "HALDER.NS",
      "SAKUMA.NS",
      "MEGASTAR.NS",
      "VIKASLIFE.NS",
      "KCPSUGIND.NS",
      "CLEDUCATE.NS",
      "UNITEDTEA.NS",
      "KMSUGAR.NS",
      "PONNIERODE.NS",
      "PKTEA.NS",
      "JAYSREETEA.NS",
      "SAKHTISUG.NS",
      "KOTARISUG.NS",
      "RANASUG.NS",
      "MKPL.NS",
      "GILLANDERS.NS",
      "ASPINWALL.NS",
      "ZEELEARN.NS",
      "DTIL.NS",
      "ESSENTIA.NS",
      "NORBTEAEXP.NS",
      "VISHWARAJ.NS",
      "GANGESSECU.NS",
      "ANIKINDS.NS",
      "COMPUSOFT.NS",
      "GROBTEA.NS",
      "RAJSREESUG.NS",
      "PALASHSECU.NS",
      "KOHINOOR.NS",
      "UMAEXPORTS.NS",
      "LCCINFOTEC.NS",
      "JHS.NS",
      "AJOONI.NS",
      "AGRITECH.NS",
      "ROML.NS",
      "FCONSUMER.NS",
      "NGIL.NS",
      "PRUDMOULI.NS",
      "RKDL.NS",
      "GOLDENTOBC.NS",
      "NKIND.NS",
      "AMBICAAGAR.NS",
      "SIMBHALS.NS",
      "TREEHOUSE.NS",
      "KEEPLEARN.NS",
      "SANWARIA.NS",
      "GTECJAINX.NS",
      "RETAIL.NS",
      "UMESLTD.NS",
      "EDUCOMP.NS",
      "SRPL.NS",
      "MTEDUCARE.NS",
      "SHANTI.NS"
    ],
    "Energy": [
      "RELIANCE.NS",
      "ONGC.NS",
      "ADANIENT.NS",
      "COALINDIA.NS",
      "IOC.NS",
      "BPCL.NS",
      "OIL.NS",
      "HINDPETRO.NS",
      "PETRONET.NS",
      "MRPL.NS",
      "AEGISLOG.NS",
      "AEGISVOPAK.NS",
      "GMDCLTD.NS",
      "CASTROLIND.NS",
      "CHENNPETRO.NS",
      "GMRP&UI.NS",
      "TRUALT.NS",
      "REFEX.NS",
      "DEEPINDS.NS",
      "PRABHA.NS",
      "ANTELOPUS.NS",
      "HINDOILEXP.NS",
      "CONFIPET.NS",
      "DOLPHIN.NS",
      "JINDRILL.NS",
      "PANAMAPET.NS",
      "ASIANENE.NS",
      "GANDHAR.NS",
      "LIKHITHA.NS",
      "SOUTHWEST.NS",
      "UNIDT.NS",
      "OILCOUNTUB.NS",
      "GULFPETRO.NS",
      "ALPHAGEO.NS",
      "ABAN.NS",
      "AAKASH.NS",
      "ANMOL.NS"
    ],
    "Financial_Services": [
      "HDFCBANK.NS",
      "SBIN.NS",
      "ICICIBANK.NS",
      "BAJFINANCE.NS",
      "LICI.NS",
      "AXISBANK.NS",
      "KOTAKBANK.NS",
      "BAJAJFINSV.NS",
      "SHRIRAMFIN.NS",
      "SBILIFE.NS",
      "ICICIAMC.NS",
      "JIOFIN.NS",
      "MUTHOOTFIN.NS",
      "PFC.NS",
      "BANKBARODA.NS",
      "UNIONBANK.NS",
      "TATACAP.NS",
      "BSE.NS",
      "HDFCLIFE.NS",
      "INDIANB.NS",
      "IRFC.NS",
      "CHOLAFIN.NS",
      "PNB.NS",
      "CANBK.NS",
      "GROWW.NS",
      "BAJAJHLDNG.NS",
      "HDFCAMC.NS",
      "RECLTD.NS",
      "ICICIGI.NS",
      "ABCAPITAL.NS",
      "IDBI.NS",
      "ICICIPRULI.NS",
      "AUBANK.NS",
      "FEDERALBNK.NS",
      "BAJAJHFL.NS",
      "MCX.NS",
      "LTF.NS",
      "GICRE.NS",
      "POLICYBZR.NS",
      "IOB.NS",
      "BANKINDIA.NS",
      "SBICARD.NS",
      "INDUSINDBK.NS",
      "YESBANK.NS",
      "NAM-INDIA.NS",
      "MFSL.NS",
      "IDFCFIRSTB.NS",
      "MAHABANK.NS",
      "SUNDARMFIN.NS",
      "HDBFS.NS",
      "MOTILALOFS.NS",
      "M&MFIN.NS",
      "360ONE.NS",
      "PIRAMALFIN.NS",
      "HUDCO.NS",
      "AIIL.NS",
      "TATAINVEST.NS",
      "IREDA.NS",
      "POONAWALLA.NS",
      "UCOBANK.NS",
      "CENTRALBK.NS",
      "ANANDRATHI.NS",
      "CRISIL.NS",
      "GODIGIT.NS",
      "CHOLAHLDNG.NS",
      "ABSLAMC.NS",
      "LICHSGFIN.NS",
      "NIACL.NS",
      "STARHEALTH.NS",
      "KARURVYSYA.NS",
      "CDSL.NS",
      "BANDHANBNK.NS",
      "ANGELONE.NS",
      "MANAPPURAM.NS",
      "NUVAMA.NS",
      "PNBHOUSING.NS",
      "AADHARHFC.NS",
      "RBLBANK.NS",
      "CREDITACC.NS",
      "IIFL.NS",
      "CUB.NS",
      "CGCL.NS",
      "SAMMAANCAP.NS",
      "PSB.NS",
      "CHOICEIN.NS",
      "IFCI.NS",
      "MAHSCOOTER.NS",
      "CANHLIFE.NS",
      "NIVABUPA.NS",
      "J&KBANK.NS",
      "JMFINANCIL.NS",
      "FIVESTAR.NS",
      "UTIAMC.NS",
      "UJJIVANSFB.NS",
      "IEX.NS",
      "HOMEFIRST.NS",
      "APTUS.NS",
      "CANFINHOME.NS",
      "EDELWEISS.NS",
      "TMB.NS",
      "PRUDENT.NS",
      "SBFC.NS",
      "AAVAS.NS",
      "SOUTHBANK.NS",
      "KTKBANK.NS",
      "IIFLCAPS.NS",
      "TSFINV.NS",
      "INDIASHLTR.NS",
      "RELIGARE.NS",
      "EQUITASBNK.NS",
      "CSBBANK.NS",
      "DCBBANK.NS",
      "MASFIN.NS",
      "FEDFINA.NS",
      "ICRA.NS",
      "CRAMC.NS",
      "PILANIINVS.NS",
      "CARERATING.NS",
      "JSFB.NS",
      "DELPHIFX.NS",
      "NORTHARC.NS",
      "INDOTHAI.NS",
      "ARSSBL.NS",
      "PAISALO.NS",
      "INDOSTAR.NS",
      "TFCILTD.NS",
      "SHAREINDIA.NS",
      "SGFIN.NS",
      "MUTHOOTMF.NS",
      "NSIL.NS",
      "AYE.NS",
      "FUSION.NS",
      "UTKARSHBNK.NS",
      "REPCOHOME.NS",
      "MONARCH.NS",
      "MUFIN.NS",
      "KICL.NS",
      "SPANDANA.NS",
      "PFS.NS",
      "SUMMITSEC.NS",
      "SATIN.NS",
      "GEOJITFSL.NS",
      "ARMANFIN.NS",
      "ALGOQUANT.NS",
      "BFINVEST.NS",
      "UGROCAP.NS",
      "SURYODAY.NS",
      "DOLATALGO.NS",
      "CENTRUM.NS",
      "PNBGILTS.NS",
      "SMCGLOBAL.NS",
      "ESAFSFB.NS",
      "CAPITALSFB.NS",
      "JPOLYINVST.NS",
      "JINDALPHOT.NS",
      "FINOPB.NS",
      "VHL.NS",
      "AFSL.NS",
      "DHANBANK.NS",
      "DAMCAPITAL.NS",
      "CIFL.NS",
      "WEALTH.NS",
      "SYSTMTXC.NS",
      "CREST.NS",
      "PRIMESECU.NS",
      "5PAISA.NS",
      "MASTERTR.NS",
      "HEXATRADEX.NS",
      "FINKURVE.NS",
      "STEL.NS",
      "DVL.NS",
      "GICHSGFIN.NS",
      "GCSL.NS",
      "ARIHANTCAP.NS",
      "BIRLAMONEY.NS",
      "BLAL.NS",
      "VLSFINANCE.NS",
      "CONSOFINVT.NS",
      "OSWALGREEN.NS",
      "LAXMIINDIA.NS",
      "EMKAY.NS",
      "MANBA.NS",
      "NBIFIN.NS",
      "THEINVEST.NS",
      "CSLFINANCE.NS",
      "GFLLIMITED.NS",
      "MONEYBOXX.NS",
      "DHUNINV.NS",
      "WELINV.NS",
      "SILINV.NS",
      "SRGHFL.NS",
      "NAHARCAP.NS",
      "AFIL.NS",
      "MANCREDIT.NS",
      "DIGISPICE.NS",
      "IITL.NS",
      "AVONMORE.NS",
      "MUTHOOTCAP.NS",
      "UYFINCORP.NS",
      "STARTECK.NS",
      "NDGL.NS",
      "ALMONDZ.NS",
      "TEAMGTY.NS",
      "SPCENET.NS",
      "IVC.NS",
      "BAIDFIN.NS",
      "CPCAP.NS",
      "KEYFINSERV.NS",
      "INDBANK.NS",
      "AUSOMENT.NS",
      "MAHAPEXLTD.NS",
      "STEELCITY.NS",
      "RHFL.NS",
      "INVENTURE.NS",
      "DBSTOCKBRO.NS",
      "BLBLIMITED.NS",
      "TRU.NS",
      "TFL.NS",
      "GATECHDVR.NS",
      "GATECH.NS",
      "HYBRIDFIN.NS",
      "ONELIFECAP.NS",
      "VIJIFIN.NS",
      "CAPTRUST.NS",
      "MASKINVEST.NS",
      "HBSL.NS",
      "LFIC.NS",
      "21STCENMGM.NS",
      "NAGREEKCAP.NS",
      "INFOMEDIA.NS",
      "WILLAMAGOR.NS",
      "TECILCHEM.NS",
      "KHANDSE.NS",
      "GLFL.NS",
      "TCIFINANCE.NS",
      "BLUECHIP.NS",
      "DCMFINSERV.NS",
      "UNIVAFOODS.NS"
    ],
    "Healthcare": [
      "SUNPHARMA.NS",
      "DIVISLAB.NS",
      "TORNTPHARM.NS",
      "APOLLOHOSP.NS",
      "LUPIN.NS",
      "DRREDDY.NS",
      "CIPLA.NS",
      "LENSKART.NS",
      "ZYDUSLIFE.NS",
      "MAXHEALTH.NS",
      "MANKIND.NS",
      "AUROPHARMA.NS",
      "FORTIS.NS",
      "ALKEM.NS",
      "GLENMARK.NS",
      "LAURUSLABS.NS",
      "BIOCON.NS",
      "ABBOTINDIA.NS",
      "ANTHEM.NS",
      "GLAXO.NS",
      "IPCALAB.NS",
      "AJANTPHARM.NS",
      "NH.NS",
      "ASTERDM.NS",
      "JBCHEPHARM.NS",
      "EMCURE.NS",
      "MEDANTA.NS",
      "GLAND.NS",
      "KIMS.NS",
      "IKS.NS",
      "LALPATHLAB.NS",
      "PFIZER.NS",
      "WOCKPHARMA.NS",
      "ASTRAZEN.NS",
      "SAILIFE.NS",
      "SAGILITY.NS",
      "PPLPHARMA.NS",
      "NATCOPHARM.NS",
      "ERIS.NS",
      "NEULANDLAB.NS",
      "ONESOURCE.NS",
      "SYNGENE.NS",
      "GRANULES.NS",
      "POLYMED.NS",
      "JUBLPHARMA.NS",
      "APLLTD.NS",
      "COHANCE.NS",
      "AGARWALEYE.NS",
      "RUBICON.NS",
      "RAINBOW.NS",
      "ALIVUS.NS",
      "CAPLIPOINT.NS",
      "INDGN.NS",
      "CONCORDBIO.NS",
      "SANOFICONR.NS",
      "MEDPLUS.NS",
      "VIJAYA.NS",
      "METROPOLIS.NS",
      "STAR.NS",
      "PARKHOSPS.NS",
      "VIYASH.NS",
      "PGHL.NS",
      "HCG.NS",
      "JSLL.NS",
      "JLHL.NS",
      "SHILPAMED.NS",
      "AKUMS.NS",
      "SANOFI.NS",
      "MARKSANS.NS",
      "YATHARTH.NS",
      "SUDEEPPHRM.NS",
      "BLUEJET.NS",
      "THYROCARE.NS",
      "AARTIPHARM.NS",
      "NEPHROPLUS.NS",
      "FDC.NS",
      "ENTERO.NS",
      "SUPRIYA.NS",
      "SUVEN.NS",
      "SPARC.NS",
      "INNOVACAP.NS",
      "ZOTA.NS",
      "SMSPHARMA.NS",
      "INDRAMEDCO.NS",
      "SENORES.NS",
      "AARTIDRUGS.NS",
      "RPGLIFE.NS",
      "GUJTHEM.NS",
      "GUFICBIO.NS",
      "ORCHPHARMA.NS",
      "BLISSGVS.NS",
      "FISCHER.NS",
      "MEDIASSIST.NS",
      "DCAL.NS",
      "IOLCP.NS",
      "UNICHEMLAB.NS",
      "SOLARA.NS",
      "HIKAL.NS",
      "MOREPENLAB.NS",
      "VIMTALABS.NS",
      "PANACEABIO.NS",
      "INDOCO.NS",
      "KRSNAA.NS",
      "SAIPARENT.NS",
      "WINDLAS.NS",
      "SHALBY.NS",
      "AMRUTANJAN.NS",
      "SURAKSHA.NS",
      "BETA.NS",
      "VENUSREM.NS",
      "NGLFINE.NS",
      "SAKAR.NS",
      "JAGSNPHARM.NS",
      "SYNCOMF.NS",
      "HESTERBIO.NS",
      "TTKHLTCARE.NS",
      "INDSWFTLAB.NS",
      "LINCOLN.NS",
      "GPTHEALTH.NS",
      "BAJAJHCARE.NS",
      "TARSONS.NS",
      "LAXMIDENTL.NS",
      "WANBURY.NS",
      "SIGACHI.NS",
      "SASTASUNDR.NS",
      "GKSL.NS",
      "MAXIND.NS",
      "ANUHPHR.NS",
      "THEMISMED.NS",
      "KOPRAN.NS",
      "AHCL.NS",
      "GAUDIUMIVF.NS",
      "TAKE.NS",
      "FABTECH.NS",
      "AMANTA.NS",
      "KILITCH.NS",
      "ALBERTDAVD.NS",
      "UNIVPHOTO.NS",
      "ZIMLAB.NS",
      "HALEOSLABS.NS",
      "BAFNAPH.NS",
      "MEDICAMEQ.NS",
      "MEDICO.NS",
      "VALIANTLAB.NS",
      "NURECA.NS",
      "NECLIFE.NS",
      "LOTUSEYE.NS",
      "BROOKS.NS",
      "LYKALABS.NS",
      "AAREYDRUGS.NS",
      "NATCAPSUQ.NS",
      "ALPA.NS",
      "KREBSBIO.NS",
      "BALAXI.NS",
      "BALPHARMA.NS",
      "PAR.NS",
      "VAISHALI.NS",
      "VIVIMEDLAB.NS",
      "BIOFILCHEM.NS",
      "MANGALAM.NS",
      "LASA.NS",
      "ORTINGLOBE.NS",
      "CORONA.NS"
    ],
    "Industrials": [
      "LT.NS",
      "ADANIPORTS.NS",
      "BEL.NS",
      "HAL.NS",
      "INDIGO.NS",
      "ABB.NS",
      "CUMMINSIND.NS",
      "POWERINDIA.NS",
      "SIEMENS.NS",
      "POLYCAB.NS",
      "CGPOWER.NS",
      "GVT&D.NS",
      "ASHOKLEY.NS",
      "MAZDOCK.NS",
      "GMRAIRPORT.NS",
      "BHEL.NS",
      "HAVELLS.NS",
      "SRF.NS",
      "SUZLON.NS",
      "RVNL.NS",
      "JSWINFRA.NS",
      "TIINDIA.NS",
      "BDL.NS",
      "SUPREMEIND.NS",
      "APARINDS.NS",
      "ASTRAL.NS",
      "THERMAX.NS",
      "KEI.NS",
      "COCHINSHIP.NS",
      "CONCOR.NS",
      "AIAENG.NS",
      "BLUESTARCO.NS",
      "3MINDIA.NS",
      "DELHIVERY.NS",
      "ESCORTS.NS",
      "GODREJIND.NS",
      "GRSE.NS",
      "TIMKEN.NS",
      "HONAUT.NS",
      "IRB.NS",
      "SCHNEIDER.NS",
      "NBCC.NS",
      "CPPLUS.NS",
      "PTCIL.NS",
      "KIRLOSENG.NS",
      "HBLENGINE.NS",
      "KPIL.NS",
      "GESHIP.NS",
      "DATAPATTNS.NS",
      "KAJARIACER.NS",
      "DCMSHRIRAM.NS",
      "NAVA.NS",
      "GRINDWELL.NS",
      "CARBORUNIV.NS",
      "ELGIEQUIP.NS",
      "JYOTICNC.NS",
      "KSB.NS",
      "RRKABEL.NS",
      "INOXWIND.NS",
      "KEC.NS",
      "JSWHL.NS",
      "TRITURBINE.NS",
      "LMW.NS",
      "DOMS.NS",
      "TDPOWERSYS.NS",
      "KIRLOSBROS.NS",
      "VGUARD.NS",
      "ZENTEC.NS",
      "MTARTECH.NS",
      "ARE&M.NS",
      "BEML.NS",
      "FINCABLES.NS",
      "TECHNOE.NS",
      "INOXINDIA.NS",
      "IRCON.NS",
      "TEGA.NS",
      "INGERRAND.NS",
      "GRAPHITE.NS",
      "HAPPYFORGE.NS",
      "ENGINERSIN.NS",
      "BLUEDART.NS",
      "BLS.NS",
      "SCI.NS",
      "GRAVITA.NS",
      "AFCONS.NS",
      "AZAD.NS",
      "JWL.NS",
      "SKFINDUS.NS",
      "HEG.NS",
      "CEMPRO.NS",
      "ACE.NS",
      "FINPIPE.NS",
      "ATLANTAELE.NS",
      "SWANCORP.NS",
      "OLECTRA.NS",
      "AEQUS.NS",
      "CYIENT.NS",
      "RITES.NS",
      "RKFORGE.NS",
      "VOLTAMP.NS",
      "NCC.NS",
      "SWANDEF.NS",
      "TITAGARH.NS",
      "VESUVIUS.NS",
      "ELECON.NS",
      "MMTC.NS",
      "APOLLO.NS",
      "TARIL.NS",
      "WABAG.NS",
      "ESABINDIA.NS",
      "GRINFRA.NS",
      "SHADOWFAX.NS",
      "KIRLPNU.NS",
      "NESCO.NS",
      "RHIM.NS",
      "SKFINDIA.NS",
      "QPOWER.NS",
      "GENUSPOWER.NS",
      "HMT.NS",
      "TCI.NS",
      "ISGEC.NS",
      "TRANSRAILL.NS",
      "DIACABS.NS",
      "GPPL.NS",
      "LLOYDSENGG.NS",
      "DBL.NS",
      "AXISCADES.NS",
      "POWERMECH.NS",
      "SGMART.NS",
      "LOTUSDEV.NS",
      "CERA.NS",
      "PRAJIND.NS",
      "PRECWIRE.NS",
      "SHAKTIPUMP.NS",
      "HNDFDS.NS",
      "LATENTVIEW.NS",
      "WELENT.NS",
      "PARAS.NS",
      "SHREEJISPG.NS",
      "BALUFORGE.NS",
      "TIIL.NS",
      "AJAXENGG.NS",
      "SHILCTECH.NS",
      "AHLUCONT.NS",
      "CEIGALL.NS",
      "POWERICA.NS",
      "ELECTCAST.NS",
      "CMSINFO.NS",
      "TVSSCS.NS",
      "PNCINFRA.NS",
      "IONEXCHANG.NS",
      "VRLLOG.NS",
      "UNIMECH.NS",
      "HCC.NS",
      "VSTTILLERS.NS",
      "OMNI.NS",
      "SIS.NS",
      "SKIPPER.NS",
      "KMEW.NS",
      "OSWALPUMPS.NS",
      "GMMPFAUDLR.NS",
      "PDSL.NS",
      "TEXRAIL.NS",
      "ANUP.NS",
      "SEAMECLTD.NS",
      "MAHLOG.NS",
      "AEROFLEX.NS",
      "MANINFRA.NS",
      "SINDHUTRAD.NS",
      "HGINFRA.NS",
      "SHANTIGEAR.NS",
      "JKIL.NS",
      "ROLEXRINGS.NS",
      "ROSSTECH.NS",
      "KSHINTL.NS",
      "GREAVESCOT.NS",
      "EIEL.NS",
      "ASHOKA.NS",
      "HARSHA.NS",
      "RAMKY.NS",
      "KNRCON.NS",
      "FLAIR.NS",
      "INTERARCH.NS",
      "GVPIL.NS",
      "RAMRAT.NS",
      "PITTIENG.NS",
      "KIRLOSIND.NS",
      "SIGMAADV.NS",
      "JISLDVREQS.NS",
      "MSTCLTD.NS",
      "HIRECT.NS",
      "BBL.NS",
      "BALMLAWRIE.NS",
      "QUESS.NS",
      "SUNCLAY.NS",
      "GATEWAY.NS",
      "SBCL.NS",
      "PSPPROJECT.NS",
      "POKARNA.NS",
      "SANGHVIMOV.NS",
      "UNIVCABLES.NS",
      "PRINCEPIPE.NS",
      "RAYMOND.NS",
      "SETL.NS",
      "MARINE.NS",
      "PATELENG.NS",
      "DEEDEV.NS",
      "DREDGECORP.NS",
      "WINDMACHIN.NS",
      "JASH.NS",
      "BGRENERGY.NS",
      "JISLJALEQS.NS",
      "RAMCOIND.NS",
      "CEINSYS.NS",
      "AWFIS.NS",
      "UNIPARTS.NS",
      "PENIND.NS",
      "ASIANTILES.NS",
      "HONDAPOWER.NS",
      "EVEREADY.NS",
      "MMFL.NS",
      "KPEL.NS",
      "HLEGLAS.NS",
      "HPL.NS",
      "ADVAIT.NS",
      "NITCO.NS",
      "APOLLOPIPE.NS",
      "CAPACITE.NS",
      "JAYKAY.NS",
      "TCIEXP.NS",
      "TEAMLEASE.NS",
      "PIXTRANS.NS",
      "DCXINDIA.NS",
      "SPECTRUM.NS",
      "HGS.NS",
      "THEJO.NS",
      "SERVOTECH.NS",
      "BFUTILITIE.NS",
      "EMSLIMITED.NS",
      "MBEL.NS",
      "EPACKPEB.NS",
      "SOMANYCERA.NS",
      "VIKRAN.NS",
      "SIMPLEXINF.NS",
      "INDIANHUME.NS",
      "JNKINDIA.NS",
      "GARUDA.NS",
      "EXICOM.NS",
      "SBC.NS",
      "TIL.NS",
      "INDOTECH.NS",
      "ADOR.NS",
      "NIBE.NS",
      "BLSE.NS",
      "DYCL.NS",
      "SPMLINFRA.NS",
      "VINDHYATEL.NS",
      "JYOTISTRUC.NS",
      "AWHCL.NS",
      "NAVKARCORP.NS",
      "VIDYAWIRES.NS",
      "GPTINFRA.NS",
      "SEPC.NS",
      "ONEPOINT.NS",
      "WENDT.NS",
      "DIGITIDE.NS",
      "KRISHNADEF.NS",
      "ALLCARGO.NS",
      "EKC.NS",
      "ALLDIGI.NS",
      "ICEMAKE.NS",
      "QUADFUTURE.NS",
      "WALCHANNAG.NS",
      "SRM.NS",
      "RAJOOENG.NS",
      "MANAKCOAT.NS",
      "BLKASHYAP.NS",
      "RMDRIP.NS",
      "NELCAST.NS",
      "RIIL.NS",
      "ALICON.NS",
      "BIRLANU.NS",
      "ROTO.NS",
      "SALZERELEC.NS",
      "DIFFNKG.NS",
      "GANDHITUBE.NS",
      "CONTROLPR.NS",
      "MACPOWER.NS",
      "ABINFRA.NS",
      "HARDWYN.NS",
      "EIMCOELECO.NS",
      "UDS.NS",
      "BLUSPRING.NS",
      "YUKEN.NS",
      "WCIL.NS",
      "MAMATA.NS",
      "GALAPREC.NS",
      "ARIS.NS",
      "GKWLIMITED.NS",
      "PVP.NS",
      "TEMBO.NS",
      "OMINFRAL.NS",
      "BAJAJINDEF.NS",
      "KAPSTON.NS",
      "PROSTARM.NS",
      "JITFINFRA.NS",
      "KOKUYOCMLN.NS",
      "STERTOOLS.NS",
      "VASCONEQ.NS",
      "KABRAEXTRU.NS",
      "ECOSMOBLTY.NS",
      "SGIL.NS",
      "KECL.NS",
      "SEJALLTD.NS",
      "ZUARIIND.NS",
      "INNOVISION.NS",
      "SUPREMEINF.NS",
      "KRYSTAL.NS",
      "DENTA.NS",
      "CCCL.NS",
      "ATL.NS",
      "BIGBLOC.NS",
      "MEIL.NS",
      "INDOFARM.NS",
      "MALLCOM.NS",
      "SVLL.NS",
      "STCINDIA.NS",
      "SNOWMAN.NS",
      "DCMSIL.NS",
      "LINC.NS",
      "GUJAPOLLO.NS",
      "ELIN.NS",
      "EVERESTIND.NS",
      "WSI.NS",
      "RITCO.NS",
      "GLOTTIS.NS",
      "ESSARSHPNG.NS",
      "VISAKAIND.NS",
      "VPRPL.NS",
      "SICALLOG.NS",
      "LOKESHMACH.NS",
      "REPRO.NS",
      "TIRUPATIFL.NS",
      "PPL.NS",
      "INTLCONV.NS",
      "TARACHAND.NS",
      "MODISONLTD.NS",
      "GICL.NS",
      "RADIANTCMS.NS",
      "KRITI.NS",
      "RMC.NS",
      "S&SPOWER.NS",
      "ORIENTBELL.NS",
      "EUROBOND.NS",
      "MBLINFRA.NS",
      "DREAMFOLKS.NS",
      "KOTHARIPRO.NS",
      "MAZDA.NS",
      "DENORA.NS",
      "NRL.NS",
      "TICL.NS",
      "RPPINFRA.NS",
      "HILINFRA.NS",
      "IL&FSENGG.NS",
      "MANAKSIA.NS",
      "MAHEPC.NS",
      "MOLDTECH.NS",
      "MARKOLINES.NS",
      "ATLANTAA.NS",
      "GAYAPROJ.NS",
      "DBEIL.NS",
      "TRANSWORLD.NS",
      "ATALREAL.NS",
      "DPWIRES.NS",
      "TIGERLOGS.NS",
      "A2ZINFRA.NS",
      "DJML.NS",
      "AARON.NS",
      "EXXARO.NS",
      "TRF.NS",
      "GEEKAYWIRE.NS",
      "GLOBECIVIL.NS",
      "APOLSINHOT.NS",
      "OMFREIGHT.NS",
      "SAHYADRI.NS",
      "NIPPOBATRY.NS",
      "JKIPL.NS",
      "GLOBALVECT.NS",
      "GENCON.NS",
      "UNIVASTU.NS",
      "PIGL.NS",
      "AVG.NS",
      "RACE.NS",
      "PRITIKAUTO.NS",
      "CORDSCABLE.NS",
      "AFFORDABLE.NS",
      "VETO.NS",
      "ROSSELLIND.NS",
      "RVTH.NS",
      "MURUDCERA.NS",
      "AARVI.NS",
      "SADBHAV.NS",
      "BRNL.NS",
      "PLAZACABLE.NS",
      "NIRAJ.NS",
      "NECCLTD.NS",
      "REPL.NS",
      "LANDSMILL.NS",
      "HECPROJECT.NS",
      "SECMARK.NS",
      "SIGIND.NS",
      "RUCHINFRA.NS",
      "AARTECH.NS",
      "CROWN.NS",
      "MAHESHWARI.NS",
      "TARMAT.NS",
      "SOMICONVEY.NS",
      "USK.NS",
      "DCM.NS",
      "TEXMOPIPES.NS",
      "HILTON.NS",
      "LATTEYS.NS",
      "MITCON.NS",
      "OBCL.NS",
      "REGENCERAM.NS",
      "DUCON.NS",
      "BCPL.NS",
      "SADBHIN.NS",
      "BEARDSELL.NS",
      "SMLT.NS",
      "RKEC.NS",
      "TOTAL.NS",
      "JETFREIGHT.NS",
      "SEMAC.NS",
      "BANKA.NS",
      "ATAM.NS",
      "ACCURACY.NS",
      "IL&FSTRANS.NS",
      "PATINTLOG.NS",
      "NIBL.NS",
      "NOIDATOLL.NS",
      "DHRUV.NS",
      "CEREBRAINT.NS",
      "TPHQ.NS",
      "AHLADA.NS",
      "GAYAHWS.NS",
      "KHAITANLTD.NS",
      "ORIENTALTL.NS",
      "TARAPUR.NS",
      "AKASH.NS",
      "LEXUS.NS",
      "MANUGRAPH.NS",
      "AROGRANITE.NS",
      "GANGAFORGE.NS",
      "AKG.NS",
      "MADHUCON.NS",
      "ARSHIYA.NS",
      "MADHAV.NS",
      "KAUSHALYA.NS",
      "KSHITIJPOL.NS",
      "SKIL.NS",
      "AURIGROW.NS",
      "ACEINTEG.NS",
      "DIL.NS",
      "MEP.NS",
      "TIJARIA.NS",
      "LAKPRE.NS",
      "PREMIER.NS",
      "SETUINFRA.NS",
      "CMICABLES.NS",
      "SANCO.NS"
    ],
    "Real_Estate": [
      "DLF.NS",
      "LODHA.NS",
      "PHOENIXLTD.NS",
      "OBEROIRLTY.NS",
      "PRESTIGE.NS",
      "GODREJPROP.NS",
      "BRIGADE.NS",
      "ANANTRAJ.NS",
      "ABREL.NS",
      "SOBHA.NS",
      "SIGNATURE.NS",
      "MAHLIFE.NS",
      "EMBDL.NS",
      "WEWORK.NS",
      "KALPATARU.NS",
      "MAXESTATES.NS",
      "DBREALTY.NS",
      "PURVA.NS",
      "RUSTOMJEE.NS",
      "GANESHHOU.NS",
      "SMARTWORKS.NS",
      "SUNTECK.NS",
      "AGIIL.NS",
      "TARC.NS",
      "HEMIPROP.NS",
      "ASHIANA.NS",
      "INDIQUBE.NS",
      "MARATHON.NS",
      "IBULLSLTD.NS",
      "KOLTEPATIL.NS",
      "RAYMONDREL.NS",
      "HUBTOWN.NS",
      "ARVSMART.NS",
      "EFCIL.NS",
      "AJMERA.NS",
      "ALEMBICLTD.NS",
      "ARKADE.NS",
      "SCILAL.NS",
      "OMAXE.NS",
      "SHRIRAMPPS.NS",
      "AURUM.NS",
      "UNITECH.NS",
      "TEXINFRA.NS",
      "ARIHANTSUP.NS",
      "SURAJEST.NS",
      "PROZONER.NS",
      "ELDEHSG.NS",
      "OSWALAGRO.NS",
      "TREL.NS",
      "GEECEE.NS",
      "PENINLAND.NS",
      "MODIS.NS",
      "SBGLP.NS",
      "NILASPACES.NS",
      "PTL.NS",
      "PANSARI.NS",
      "PARSVNATH.NS",
      "DEVX.NS",
      "NILAINFRA.NS",
      "EMAMIREAL.NS",
      "MODIRUBBER.NS",
      "ASHIMASYN.NS",
      "RVHL.NS",
      "SHRADHA.NS",
      "SUPREME.NS",
      "SUMIT.NS",
      "LANCORHOL.NS",
      "AMJLAND.NS",
      "UMIYA-MRO.NS",
      "PRAENG.NS",
      "VIPULLTD.NS",
      "CORALFINAC.NS",
      "ARTNIRMAN.NS",
      "HDIL.NS",
      "LPDC.NS",
      "MOTOGENFIN.NS",
      "DHARAN.NS",
      "ICDSLTD.NS",
      "FMNL.NS",
      "ANSALAPI.NS",
      "COUNCODOS.NS",
      "ROLLT.NS",
      "EUROTEXIND.NS"
    ],
    "Technology": [
      "TCS.NS",
      "INFY.NS",
      "HCLTECH.NS",
      "WIPRO.NS",
      "LTM.NS",
      "TECHM.NS",
      "LGEINDIA.NS",
      "WAAREEENER.NS",
      "PERSISTENT.NS",
      "PAYTM.NS",
      "DIXON.NS",
      "OFSS.NS",
      "MPHASIS.NS",
      "PREMIERENE.NS",
      "COFORGE.NS",
      "LTTS.NS",
      "HEXT.NS",
      "ITI.NS",
      "TATAELXSI.NS",
      "KAYNES.NS",
      "TATATECH.NS",
      "URBANCO.NS",
      "PINELABS.NS",
      "KPITTECH.NS",
      "NETWEB.NS",
      "CAMS.NS",
      "REDINGTON.NS",
      "SYRMA.NS",
      "EMMVEE.NS",
      "KFINTECH.NS",
      "FSL.NS",
      "FRACTAL.NS",
      "PGEL.NS",
      "ECLERX.NS",
      "HFCL.NS",
      "ZENSARTECH.NS",
      "STLTECH.NS",
      "BLACKBUCK.NS",
      "BSOFT.NS",
      "ASTRAMICRO.NS",
      "BBOX.NS",
      "INTELLECT.NS",
      "VIKRAMSOLR.NS",
      "TEJASNET.NS",
      "AMAGI.NS",
      "AVALON.NS",
      "UTLSOLAR.NS",
      "SONATSOFTW.NS",
      "RATEGAIN.NS",
      "BORORENEW.NS",
      "63MOONS.NS",
      "KRN.NS",
      "NEWGEN.NS",
      "TANLA.NS",
      "SAATVIKGL.NS",
      "HAPPSTMNDS.NS",
      "CCAVENUE.NS",
      "MAPMYINDIA.NS",
      "E2E.NS",
      "MASTEK.NS",
      "AURIONPRO.NS",
      "EBGNG.NS",
      "CAPILLARY.NS",
      "SWSOLAR.NS",
      "CENTUM.NS",
      "DATAMATICS.NS",
      "AVANTEL.NS",
      "EMUDHRA.NS",
      "STYL.NS",
      "WEBELSOLAR.NS",
      "PACEDIGITK.NS",
      "ZAGGLE.NS",
      "MOSCHIP.NS",
      "CIGNITITEC.NS",
      "OPTIEMUS.NS",
      "RPSGVENT.NS",
      "RSYSTEMS.NS",
      "RPTECH.NS",
      "NPST.NS",
      "CYIENTDLM.NS",
      "RELTD.NS",
      "UEL.NS",
      "PROTEAN.NS",
      "NUCLEUS.NS",
      "KERNEX.NS",
      "IDEAFORGE.NS",
      "SASKEN.NS",
      "WAAREEINDO.NS",
      "RISHABH.NS",
      "INFOBEAN.NS",
      "SAKSOFT.NS",
      "ACCELYA.NS",
      "RAMCOSYS.NS",
      "SOLARWORLD.NS",
      "SILVERTUC.NS",
      "MCLOUD.NS",
      "GTLINFRA.NS",
      "DLINKINDIA.NS",
      "MOBIKWIK.NS",
      "NELCO.NS",
      "DSSL.NS",
      "SOLEX.NS",
      "IVALUE.NS",
      "ORIENTTECH.NS",
      "EXPLEOSOL.NS",
      "PARACABLES.NS",
      "IKIO.NS",
      "GENESYS.NS",
      "IZMO.NS",
      "UNIECOM.NS",
      "EXCELSOFT.NS",
      "MICEL.NS",
      "SWELECTES.NS",
      "CNL.NS",
      "QUICKHEAL.NS",
      "KELLTONTEC.NS",
      "INNOVANA.NS",
      "TVSELECT.NS",
      "NINSYS.NS",
      "KSOLVES.NS",
      "VAKRANGEE.NS",
      "XCHANGING.NS",
      "ADSL.NS",
      "MINDTECK.NS",
      "URJA.NS",
      "ONWARDTEC.NS",
      "DCI.NS",
      "SUBEXLTD.NS",
      "IRIS.NS",
      "SIGMA.NS",
      "TERASOFT.NS",
      "PANACHE.NS",
      "BIRLACABLE.NS",
      "SOFTTECH.NS",
      "TREJHARA.NS",
      "HCL-INSYS.NS",
      "XTGLOBAL.NS",
      "CYBERTECH.NS",
      "VERTOZ.NS",
      "KAVDEFENCE.NS",
      "TRACXN.NS",
      "INSPIRISYS.NS",
      "3IINFOLTD.NS",
      "FCSSOFT.NS",
      "ASMS.NS",
      "INTENTECH.NS",
      "BPL.NS",
      "ALANKIT.NS",
      "DRCSYSTEMS.NS",
      "GOLDTECH.NS",
      "NITIRAJ.NS",
      "AIRAN.NS",
      "VIRINCHI.NS",
      "XELPMOC.NS",
      "DEVIT.NS",
      "TRIGYN.NS",
      "VLEGOV.NS",
      "EQUIPPP.NS",
      "AAATECH.NS",
      "RELIABLE.NS",
      "SMARTLINK.NS",
      "GVPTECH.NS",
      "SURANASOL.NS",
      "CURAA.NS",
      "GENSOL.NS",
      "AKSHOPTFBR.NS",
      "SECURKLOUD.NS",
      "RSSOFTWARE.NS",
      "CALSOFT.NS",
      "HGM.NS",
      "SUVIDHAA.NS",
      "ORCHASP.NS",
      "CTE.NS",
      "ADROITINFO.NS",
      "WEWIN.NS",
      "TNTELE.NS",
      "PALREDTEC.NS",
      "GSS.NS",
      "AGSTRA.NS",
      "SHYAMTEL.NS",
      "COMPINFO.NS",
      "BGLOBAL.NS",
      "QUINTEGRA.NS"
    ],
    "Utilities": [
      "NTPC.NS",
      "ADANIPOWER.NS",
      "POWERGRID.NS",
      "ADANIGREEN.NS",
      "ADANIENSOL.NS",
      "TATAPOWER.NS",
      "ENRIN.NS",
      "GAIL.NS",
      "JSWENERGY.NS",
      "NTPCGREEN.NS",
      "NHPC.NS",
      "TORNTPOWER.NS",
      "ATGL.NS",
      "NLCINDIA.NS",
      "SJVN.NS",
      "IGL.NS",
      "GUJGASLTD.NS",
      "CESC.NS",
      "ACMESOLAR.NS",
      "GSPL.NS",
      "JPPOWER.NS",
      "RPOWER.NS",
      "MGL.NS",
      "CLEANMAX.NS",
      "WAAREERTL.NS",
      "KPIGREEN.NS",
      "INOXGREEN.NS",
      "PTC.NS",
      "RTNPOWER.NS",
      "RELINFRA.NS",
      "GKENERGY.NS",
      "GIPCL.NS",
      "BAJEL.NS",
      "GREENPOWER.NS",
      "DPSCLTD.NS",
      "IRMENERGY.NS",
      "CEWATER.NS",
      "GVKPIL.NS",
      "SURANAT&P.NS",
      "INDOWIND.NS",
      "ENERGYDEV.NS",
      "KARMAENG.NS"
    ]
  }

# Nifty 50 constituents for get_market_map_tickers()
LARGE_CAPS =  [
    "RELIANCE.NS",
    "HDFCBANK.NS","BHARTIARTL.NS","SBIN.NS","ICICIBANK.NS","TCS.NS","BAJFINANCE.NS","LT.NS","INFY.NS","LICI.NS","HINDUNILVR.NS","AXISBANK.NS",""
    "MARUTI.NS","SUNPHARMA.NS","TITAN.NS",
    "HCLTECH.NS","M&M.NS","NTPC.NS","ITC.NS","KOTAKBANK.NS","ONGC.NS","ADANIPOWER.NS","ULTRACEMCO.NS","ADANIPORTS.NS",
    "BEL.NS","JSWSTEEL.NS","VEDL.NS","DMART.NS","ADANIENT.NS","BAJAJFINSV.NS","POWERGRID.NS","HAL.NS","BAJAJ-AUTO.NS","COALINDIA.NS",
    "TATASTEEL.NS","HINDZINC.NS","NESTLEIND.NS","SHRIRAMFIN.NS","ASIANPAINT.NS","HINDALCO.NS",
    "ETERNAL.NS","WIPRO.NS","IOC.NS","EICHERMOT.NS","SBILIFE.NS","GRASIM.NS","ADANIGREEN.NS","TVSMOTOR.NS","INDIGO.NS","ICICIAMC.NS"
  ]
# ── Cache ──────────────────────────────────────────────────────────────────────

EMPTY_OHLCV = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
_cache: Dict[str, dict] = {}
_batch_cache: Dict[tuple, dict] = {}
_meta_cache: Dict[str, dict] = {}
_cache_settings = market_cache_settings()
_disk_cache = PersistentTTLCache(_cache_settings["dir"]) if _cache_settings["enabled"] else None
_price_cache_ttl = _cache_settings["history_ttl_minutes"] * 60
_batch_cache_ttl = _cache_settings["batch_ttl_minutes"] * 60
_meta_cache_ttl = _cache_settings["metadata_ttl_minutes"] * 60
_options_cache_ttl = _cache_settings["options_ttl_minutes"] * 60
_vix_cache_ttl = _cache_settings["vix_ttl_minutes"] * 60


def _empty_ohlcv() -> pd.DataFrame:
    return EMPTY_OHLCV.copy()


def _get_cached_value(store: Dict, key, ttl: int):
    now = time.time()
    entry = store.get(key)
    if entry and (now - entry['ts']) < ttl:
        return entry.get('value')
    return None


def _set_cached_value(store: Dict, key, value) -> None:
    store[key] = {'value': value, 'ts': time.time()}


def _disk_get(key: str, ttl: int):
    if _disk_cache is None:
        return None
    return _disk_cache.get(key, ttl_seconds=max(int(ttl), 0))


def _disk_set(key: str, value) -> None:
    if _disk_cache is not None:
        _disk_cache.set(key, value)


def _normalize_ohlcv(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_ohlcv()

    normalized = df.copy()
    normalized.index = pd.to_datetime(normalized.index)
    if getattr(normalized.index, 'tz', None) is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized = normalized[~normalized.index.duplicated(keep='last')].sort_index()
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in normalized.columns]
    if not cols:
        return _empty_ohlcv()
    return normalized[cols]


def cached_fetch(ticker: str, period: str, interval: str, ttl: int = 300) -> pd.DataFrame:
    key = f"{ticker}_{period}_{interval}"
    cached_data = _get_cached_value(_cache, key, ttl)
    if isinstance(cached_data, pd.DataFrame):
        return cached_data

    disk_key = f"swarm::price::{ticker.upper()}::{period}::{interval}"
    cached_data = _disk_get(disk_key, ttl)
    if isinstance(cached_data, pd.DataFrame):
        _set_cached_value(_cache, key, cached_data)
        return cached_data

    data = fetch_price_data(ticker, period, interval)
    if data is None:
        data = _empty_ohlcv()

    _set_cached_value(_cache, key, data)
    _disk_set(disk_key, data)
    return data


# ── Core Fetchers ──────────────────────────────────────────────────────────────

def fetch_price_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval, auto_adjust=True)
    except Exception:
        return _empty_ohlcv()
    return _normalize_ohlcv(df)


def fetch_multi_ticker(tickers: List[str], period: str = "2y",
                       interval: str = "1d") -> pd.DataFrame:
    try:
        data = yf.download(tickers, period=period, interval=interval,
                           group_by='ticker', auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if isinstance(data.index, pd.DatetimeIndex) and getattr(data.index, 'tz', None) is not None:
        data.index = data.index.tz_localize(None)
    return data.sort_index()


def fetch_close_matrix(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    unique_tickers = list(dict.fromkeys(tickers))
    if not unique_tickers:
        return pd.DataFrame()

    cache_key = ('close_matrix', tuple(sorted(unique_tickers)), period)
    cached = _get_cached_value(_batch_cache, cache_key, ttl=_batch_cache_ttl)
    if isinstance(cached, pd.DataFrame):
        return cached

    disk_key = f"swarm::close_matrix::{period}::{'|'.join(sorted(unique_tickers))}"
    cached = _disk_get(disk_key, _batch_cache_ttl)
    if isinstance(cached, pd.DataFrame):
        _set_cached_value(_batch_cache, cache_key, cached)
        return cached

    try:
        data = yf.download(unique_tickers, period=period, auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.index, pd.DatetimeIndex) and getattr(data.index, 'tz', None) is not None:
        data.index = data.index.tz_localize(None)

    if 'Close' in data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else 'Close' in data.columns:
        if isinstance(data.columns, pd.MultiIndex):
            closes = data['Close']
        else:
            closes = data[['Close']]
            closes.columns = unique_tickers[:1]
    else:
        closes = data
    closes = closes.dropna(how='all').sort_index()
    _set_cached_value(_batch_cache, cache_key, closes)
    _disk_set(disk_key, closes)
    return closes


# ── Options Data ───────────────────────────────────────────────────────────────

def fetch_options_chain(ticker: str, max_expirations: int = 4) -> dict:
    cache_key = f"options_chain::{ticker}::{max_expirations}"
    cached = _get_cached_value(_meta_cache, cache_key, ttl=_options_cache_ttl)
    if isinstance(cached, dict):
        return cached

    disk_key = f"swarm::{cache_key}"
    cached = _disk_get(disk_key, _options_cache_ttl)
    if isinstance(cached, dict):
        _set_cached_value(_meta_cache, cache_key, cached)
        return cached

    t = yf.Ticker(ticker)
    try:
        expirations = t.options
    except Exception:
        return {}
    chains = {}
    for exp in expirations[:max_expirations]:
        try:
            chain = t.option_chain(exp)
            chains[exp] = {'calls': chain.calls, 'puts': chain.puts}
        except Exception:
            continue
    _set_cached_value(_meta_cache, cache_key, chains)
    _disk_set(disk_key, chains)
    return chains


def get_options_flow_signals(ticker: str) -> dict:
    cache_key = f"options_flow::{ticker}"
    cached = _get_cached_value(_meta_cache, cache_key, ttl=_options_cache_ttl)
    if isinstance(cached, dict):
        return cached

    disk_key = f"swarm::{cache_key}"
    cached = _disk_get(disk_key, _options_cache_ttl)
    if isinstance(cached, dict):
        _set_cached_value(_meta_cache, cache_key, cached)
        return cached

    default_result = {'put_call_ratio': 1.0, 'unusual_calls': [], 'unusual_puts': [],
                      'implied_vol': 0.3}
    t = yf.Ticker(ticker)
    try:
        expirations = t.options
        if not expirations:
            return default_result
        chain = t.option_chain(expirations[0])
    except Exception:
        return default_result
    calls, puts = chain.calls, chain.puts
    call_vol = calls['volume'].sum() if 'volume' in calls.columns else 1
    put_vol = puts['volume'].sum() if 'volume' in puts.columns else 1
    pc_ratio = put_vol / (call_vol + 1e-9)

    unusual_calls, unusual_puts = [], []
    if all(col in calls.columns for col in ['strike', 'volume', 'openInterest']):
        calls_copy = calls.copy()
        calls_copy['vol_oi'] = calls_copy['volume'] / (calls_copy['openInterest'] + 1)
        call_cols = [c for c in ['strike', 'volume', 'openInterest', 'impliedVolatility'] if c in calls_copy.columns]
        unusual_calls = calls_copy[calls_copy['vol_oi'] > 2][call_cols].to_dict('records')
    if all(col in puts.columns for col in ['strike', 'volume', 'openInterest']):
        puts_copy = puts.copy()
        puts_copy['vol_oi'] = puts_copy['volume'] / (puts_copy['openInterest'] + 1)
        put_cols = [c for c in ['strike', 'volume', 'openInterest', 'impliedVolatility'] if c in puts_copy.columns]
        unusual_puts = puts_copy[puts_copy['vol_oi'] > 2][put_cols].to_dict('records')

    iv = calls['impliedVolatility'].mean() if 'impliedVolatility' in calls.columns else 0.3
    result = {
        'put_call_ratio': float(pc_ratio),
        'unusual_calls': unusual_calls[:10],
        'unusual_puts': unusual_puts[:10],
        'implied_vol': float(iv),
    }
    _set_cached_value(_meta_cache, cache_key, result)
    _disk_set(disk_key, result)
    return result


# ── VIX & Noise ───────────────────────────────────────────────────────────────

def get_vix_level() -> float:
    """Fetch India VIX (^INDIAVIX). Falls back to 20 if unavailable."""
    cached = _get_cached_value(_meta_cache, 'india_vix', ttl=_vix_cache_ttl)
    if cached is not None:
        return float(cached)

    disk_key = "swarm::india_vix"
    cached = _disk_get(disk_key, _vix_cache_ttl)
    if cached is not None:
        _set_cached_value(_meta_cache, 'india_vix', float(cached))
        return float(cached)

    try:
        vix = yf.Ticker('^INDIAVIX')
        hist = vix.history(period='5d')
        if not hist.empty:
            level = float(hist['Close'].iloc[-1])
            _set_cached_value(_meta_cache, 'india_vix', level)
            _disk_set(disk_key, level)
            return level
    except Exception:
        pass
    # Secondary fallback: compute realised vol of Nifty 50 as VIX proxy
    try:
        nifty = yf.Ticker('^NSEI')
        hist = nifty.history(period='30d')
        if not hist.empty:
            returns = hist['Close'].pct_change().dropna()
            level = float(returns.std() * (252 ** 0.5) * 100)
            _set_cached_value(_meta_cache, 'india_vix', level)
            _disk_set(disk_key, level)
            return level
    except Exception:
        pass
    return 20.0


def compute_noise_from_vix(vix_level: float) -> float:
    return float(np.clip(vix_level / 100.0, 0.05, 0.5))


# ── Feature Engineering ───────────────────────────────────────────────────────

def compute_swarm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute high-variance, non-redundant features actually consumed by agents.

    Variance analysis (2y data, 10 tickers) showed 9 of the original 15 features
    were redundant (r > 0.85 collinearity) or unused by any agent. Pruned to 6.
    """
    if df.empty or len(df) < 30:
        return df
    features = df.copy()

    # Momentum — used by Boids, Vicsek, ACO, Leader, Topological, Force Map, Heatmap
    features['momentum_1d'] = df['Close'].pct_change(1)
    features['momentum_5d'] = df['Close'].pct_change(5)
    features['momentum_20d'] = df['Close'].pct_change(20)

    # Volume z-score — ACO pheromone proxy, Leader detection (CV=1.33, highest)
    vol_mean = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std()
    features['volume_zscore'] = (df['Volume'] - vol_mean) / (vol_std + 1e-9)

    # RSI — ACO heuristic, Force Map quadrants
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    features['rsi'] = 100 - (100 / (1 + rs))

    # MACD histogram — ACO heuristic (difference signal, less collinear than raw MACD)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    features['macd_hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    return features.dropna()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — used for stop sizing, not stored in features cache."""
    # Forward-fill then back-fill so any intraday NaN rows don't poison the series
    high = df['High'].ffill().bfill()
    low  = df['Low'].ffill().bfill()
    close = df['Close'].ffill().bfill()
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def compute_support_resistance(
    df: pd.DataFrame,
    window: int = 20,
    n_levels: int = 3,
) -> Dict[str, List[float]]:
    """
    Detect support and resistance levels using rolling swing highs/lows.
    Clusters nearby levels within 1.5% of each other.

    Returns:
        {'support': [...], 'resistance': [...]}
        support  sorted descending (nearest below current price first)
        resistance sorted ascending (nearest above current price first)
    """
    if df.empty or len(df) < window * 2:
        return {'support': [], 'resistance': []}

    close = df['Close'].ffill().bfill()
    high  = (df['High'].ffill().bfill()  if 'High' in df.columns else close)
    low   = (df['Low'].ffill().bfill()   if 'Low'  in df.columns else close)
    current = float(close.dropna().iloc[-1])

    swing_highs = high[(high == high.rolling(window, center=True).max())].dropna()
    swing_lows  = low[ (low  == low.rolling(window, center=True).min())].dropna()

    def _cluster(levels: pd.Series, n: int) -> List[float]:
        if levels.empty:
            return []
        vals = sorted(levels.values)
        clusters, group = [], [vals[0]]
        for v in vals[1:]:
            if (v - group[0]) / (group[0] + 1e-9) < 0.015:
                group.append(v)
            else:
                clusters.append(float(np.mean(group)))
                group = [v]
        clusters.append(float(np.mean(group)))
        return clusters[:n]

    supports    = sorted([v for v in _cluster(swing_lows,  n_levels * 2) if v < current], reverse=True)[:n_levels]
    resistances = sorted([v for v in _cluster(swing_highs, n_levels * 2) if v > current])[:n_levels]
    return {'support': supports, 'resistance': resistances}


def compute_trade_levels(
    df: pd.DataFrame,
    direction: float,
    atr_multiplier: float = 1.5,
    rr_min: float = 1.5,
) -> Dict:
    """
    Compute concrete trade plan from OHLCV data and swarm direction signal.

    Entry:   nearest support (long) or resistance (short) within 2 ATR,
             else current price (breakout mode)
    Stop:    entry ± atr_multiplier * ATR14
    Target:  nearest resistance (long) or support (short);
             fallback = Fibonacci extension when no level found or R:R < rr_min
    Hold:    price distance to target / ATR per day
    """
    if df.empty or len(df) < 30:
        return {}

    atr_series = compute_atr(df)
    atr_clean  = atr_series.dropna()
    if atr_clean.empty:
        return {}

    atr          = float(atr_clean.iloc[-1])
    close_clean  = df['Close'].dropna()
    current      = float(close_clean.iloc[-1]) if not close_clean.empty else float('nan')
    if atr <= 0 or current <= 0 or np.isnan(current):
        return {}

    levels      = compute_support_resistance(df, window=20, n_levels=3)
    supports    = levels['support']
    resistances = levels['resistance']
    is_long     = direction >= 0

    # --- Entry ---
    if is_long:
        near = [s for s in supports if (current - s) <= 2 * atr]
        entry_price = near[0] if near else current
        entry_type  = 'support' if near else 'breakout'
    else:
        near = [r for r in resistances if (r - current) <= 2 * atr]
        entry_price = near[0] if near else current
        entry_type  = 'resistance' if near else 'breakout'

    # --- Stop ---
    stop_loss    = entry_price - atr_multiplier * atr if is_long else entry_price + atr_multiplier * atr
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit < 1e-6:
        return {}

    # --- Target ---
    recent_high  = float(df['High'].tail(60).max())
    recent_low   = float(df['Low'].tail(60).min())
    swing_range  = recent_high - recent_low
    fib_ext_long  = entry_price + swing_range * 0.5
    fib_ext_short = entry_price - swing_range * 0.5

    if is_long:
        take_profit = resistances[0] if resistances else fib_ext_long
        if (take_profit - entry_price) / risk_per_unit < rr_min:
            take_profit = max(resistances[-1] if len(resistances) > 1 else 0, fib_ext_long)
    else:
        take_profit = supports[0] if supports else fib_ext_short
        if (entry_price - take_profit) / risk_per_unit < rr_min:
            take_profit = min(supports[-1] if len(supports) > 1 else 9999, fib_ext_short)

    reward      = abs(take_profit - entry_price)
    risk_reward = reward / risk_per_unit
    hold_days   = max(1, min(90, int(round(reward / atr))))

    return {
        'entry_price':        round(entry_price, 2),
        'stop_loss':          round(stop_loss, 2),
        'take_profit':        round(take_profit, 2),
        'risk_reward':        round(risk_reward, 2),
        'estimated_hold_days': hold_days,
        'atr':                round(atr, 4),
        'entry_type':         entry_type,
        'levels':             levels,
    }


def compute_sector_score(
    etf_features: pd.DataFrame,
    pheromone_strength: float,
) -> float:
    """
    Composite sector attractiveness score [0, 1].
    Weights: momentum=0.40, volume=0.20, macd=0.20, pheromone=0.20
    """
    if etf_features is None or etf_features.empty:
        return 0.0

    row = etf_features.iloc[-1]

    mom = (row.get('momentum_1d', 0.0) * 0.2 +
           row.get('momentum_5d', 0.0) * 0.5 +
           row.get('momentum_20d', 0.0) * 0.3)
    mom_score  = float(np.clip(mom / 0.10, -1.0, 1.0)) * 0.5 + 0.5
    vol_score  = float(np.clip(row.get('volume_zscore', 0.0) / 3.0, -1.0, 1.0)) * 0.5 + 0.5
    macd_score = float(np.clip(row.get('macd_hist', 0.0) * 50, -1.0, 1.0)) * 0.5 + 0.5
    phe_score  = pheromone_strength / (1.0 + pheromone_strength)

    return float(0.40 * mom_score + 0.20 * vol_score + 0.20 * macd_score + 0.20 * phe_score)


def compute_correlation_matrix(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    closes = fetch_close_matrix(tickers, period)
    if closes.empty or len(closes) < 20:
        return pd.DataFrame()
    returns = closes.pct_change().dropna()
    return returns.corr()


def get_topological_neighbors(corr_matrix: pd.DataFrame,
                               n_neighbors: int = 7) -> Dict[str, List[str]]:
    """Ballerini 2008: 7 nearest neighbors by correlation RANK, not distance."""
    neighbors = {}
    for asset in corr_matrix.columns:
        ranked = corr_matrix[asset].drop(asset, errors='ignore').abs().nlargest(n_neighbors)
        neighbors[asset] = ranked.index.tolist()
    return neighbors


def _fetch_and_compute(ticker: str, period: str) -> Tuple[str, Optional[pd.DataFrame]]:
    """Fetch price data and compute swarm features for a single ticker (thread-safe)."""
    try:
        df = cached_fetch(ticker, period, "1d")
        if df.empty or len(df) < 30:
            return ticker, None
        features = compute_swarm_features(df)
        return ticker, features
    except Exception:
        return ticker, None


def fetch_all_features_parallel(
    tickers: List[str], period: str = "2y", max_workers: int = 12
) -> Dict[str, pd.DataFrame]:
    """Fetch and compute swarm features for all tickers concurrently using ThreadPoolExecutor."""
    results: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_and_compute, ticker, period): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker, features = future.result()
            if features is not None and not features.empty:
                results[ticker] = features
    return results


def get_market_map_tickers(n_stocks: int = 50) -> List[str]:
    return LARGE_CAPS[:n_stocks]


def get_all_sector_tickers() -> Dict[str, List[str]]:
    """Return the full SECTOR_STOCKS dictionary."""
    return SECTOR_STOCKS


def get_sector_for_ticker(ticker: str) -> str:
    """Reverse-lookup: given a ticker, return its sector name."""
    for sector, tickers in SECTOR_STOCKS.items():
        if ticker in tickers:
            return sector
    return "Unknown"


def get_all_tickers_flat() -> List[str]:
    """Return all tickers across all sectors as a flat, deduplicated list."""
    seen: set = set()
    result: List[str] = []
    for stocks in SECTOR_STOCKS.values():
        for s in stocks:
            if s not in seen:
                result.append(s)
                seen.add(s)
    return result


def get_sector_stock_count() -> Dict[str, int]:
    """Return count of stocks per sector."""
    return {sector: len(stocks) for sector, stocks in SECTOR_STOCKS.items()}


def fetch_features_in_batches(
    tickers: List[str],
    period: str = "2y",
    batch_size: int = 50,
    max_workers: int = 12,
    progress_callback=None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch and compute swarm features for a large ticker list in batches.

    Args:
        tickers: Full list of tickers to process.
        period: yfinance period string.
        batch_size: Number of tickers per batch (keeps yfinance happy).
        max_workers: Thread pool size per batch.
        progress_callback: Optional callable(batch_num, total_batches, n_success)
                           invoked after each batch completes.

    Returns:
        Dict mapping ticker -> features DataFrame for all successfully fetched tickers.
    """
    all_results: Dict[str, pd.DataFrame] = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(tickers))
        batch = tickers[start:end]

        batch_results = fetch_all_features_parallel(batch, period=period, max_workers=max_workers)
        all_results.update(batch_results)

        if progress_callback:
            progress_callback(batch_num + 1, total_batches, len(all_results))

        # Small delay between batches to avoid yfinance throttling
        if batch_num < total_batches - 1:
            time.sleep(0.5)

    return all_results


def get_ticker_info(ticker: str) -> dict:
    cache_key = f"ticker_info::{ticker}"
    cached = _get_cached_value(_meta_cache, cache_key, ttl=_meta_cache_ttl)
    if isinstance(cached, dict):
        return cached

    disk_key = f"swarm::{cache_key}"
    cached = _disk_get(disk_key, _meta_cache_ttl)
    if isinstance(cached, dict):
        _set_cached_value(_meta_cache, cache_key, cached)
        return cached

    try:
        t = yf.Ticker(ticker)
        info = t.info
        result = {
            'sector': info.get('sector', 'Unknown'),
            'marketCap': info.get('marketCap', 0),
            'beta': info.get('beta', 1.0),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', None),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', None),
        }
        _set_cached_value(_meta_cache, cache_key, result)
        _disk_set(disk_key, result)
        return result
    except Exception:
        return {'sector': 'Unknown', 'marketCap': 0, 'beta': 1.0,
                'fiftyTwoWeekHigh': None, 'fiftyTwoWeekLow': None}


def clear_market_cache(persist: bool = True) -> None:
    """Clear swarm in-memory caches and optionally the shared disk cache."""
    _cache.clear()
    _batch_cache.clear()
    _meta_cache.clear()
    if persist and _disk_cache is not None:
        _disk_cache.clear()


def market_cache_stats() -> dict:
    """Return shared market-cache stats for UI/debugging."""
    stats = _disk_cache.stats() if _disk_cache is not None else {
        "enabled": False,
        "path": None,
        "entry_count": 0,
        "size_bytes": 0,
        "size_mb": 0.0,
    }
    stats["memory_entries"] = len(_cache) + len(_batch_cache) + len(_meta_cache)
    return stats
