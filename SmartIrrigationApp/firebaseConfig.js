import { initializeApp, getApps } from 'firebase/app';
import { getDatabase } from 'firebase/database';
// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAngPDJAfiWmjhbbm8H2urrpXI5tn2HB30",
  authDomain: "AIzaSyAngPDJAfiWmjhbbm8H2urrpXI5tn2HB30",
  projectId: "smartirrigation-6956d",
  storageBucket: "smartirrigation-6956d.firebasestorage.app",
  messagingSenderId: "378876400494",
  appId: "1:378876400494:web:b912d6692c971718c3277f",
  databaseURL: "https://smartirrigation-6956d-default-rtdb.asia-southeast1.firebasedatabase.app/"
};

// Initialize Firebase if it hasn’t been initialized already
let firebaseApp;
if (!getApps().length) {
  firebaseApp = initializeApp(firebaseConfig);
} else {
  firebaseApp = getApps()[0];
}

// Initialize Firestore
const db = getDatabase(firebaseApp)

export { db };