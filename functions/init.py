from firebase_admin import firestore

from admobssv import AdMobSSVVerifier

db = firestore.client()
ssv_verifier = AdMobSSVVerifier()
