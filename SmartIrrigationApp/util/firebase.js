import { ref, set, get } from 'firebase/database';
import { db } from '../firebaseConfig';

// Function to add a new farm to the Firebase Realtime Database using farmName as the key
export async function addFarm(farmData) {
  try {
    const { farmName } = farmData;
    const farmRef = ref(db, `farms/${farmName}`);
    await set(farmRef, farmData);
    console.log("Farm added successfully:", farmData);
  } catch (error) {
    console.error("Error adding farm:", error);
    throw error;
  }
}

// Function to get details of a specific farm or all farms if no farmName is specified
export async function getFarmDetails(farmName = null) {
  try {
    const farmRef = farmName ? ref(db, `farms/${farmName}`) : ref(db, 'farms');
    const snapshot = await get(farmRef);
    if (snapshot.exists()) {
      return snapshot.val();
    } else {
      console.warn(farmName ? `Farm with name "${farmName}" not found.` : "No farms found.");
      return null;
    }
  } catch (error) {
    console.error("Error fetching farm details:", error);
    throw error;
  }
}
