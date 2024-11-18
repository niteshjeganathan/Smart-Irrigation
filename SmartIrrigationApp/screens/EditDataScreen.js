import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Alert, Keyboard, ScrollView, TouchableWithoutFeedback } from 'react-native';
import { getDatabase, ref, set, get } from 'firebase/database';

const EditDataScreen = ({ route, navigation }) => {
  const { farmName, selectedDate } = route.params;

  const [weatherData, setWeatherData] = useState({
    T_max: '',
    T_min: '',
    Humidity_1: '',
    Humidity_2: '',
    WindSpeed: '',
    Sunshine_Hours: '',
  });

  const labels = {
    T_max: 'Max Temperature',
    T_min: 'Min Temperature',
    Humidity_1: 'Max Humidity',
    Humidity_2: 'Min Humidity',
    WindSpeed: 'Wind Speed',
    Sunshine_Hours: 'Sunshine Hours',
  };

  useEffect(() => {
    const fetchData = async () => {
      const database = getDatabase();
      const dataRef = ref(database, `farms/${farmName}/dates/${selectedDate}`);
      const snapshot = await get(dataRef);

      if (snapshot.exists()) {
        setWeatherData(snapshot.val());
      }
    };

    fetchData();
  }, [farmName, selectedDate]);

  const handleSave = async () => {
    try {
      const database = getDatabase();
      const dataRef = ref(database, `farms/${farmName}/dates/${selectedDate}`);

      await set(dataRef, weatherData);

      Alert.alert("Success", "Data saved successfully.");
      navigation.goBack();
    } catch (error) {
      console.error("Error saving data:", error);
      Alert.alert("Error", "Failed to save data. Please try again.");
    }
  };

  const handleChange = (key, value) => {
    setWeatherData((prevData) => ({ ...prevData, [key]: value }));
  };

  return (
    <TouchableWithoutFeedback onPress={() => Keyboard.dismiss()}>
      <ScrollView
        contentContainerStyle={styles.container}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={styles.title}>Edit Data for {selectedDate}</Text>
        {Object.keys(weatherData).map((key) => (
          <View key={key} style={styles.inputContainer}>
            <Text style={styles.label}>{labels[key]}:</Text>
            <TextInput
              style={styles.input}
              value={String(weatherData[key])}  // Ensure value is a string for TextInput
              onChangeText={(value) => handleChange(key, value)}
              keyboardType="numeric"
            />
          </View>
        ))}
        <Button title="Save Data" onPress={handleSave} />
      </ScrollView>
    </TouchableWithoutFeedback>
  );
};

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 16,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  inputContainer: {
    marginBottom: 12,
  },
  label: {
    fontSize: 16,
    marginBottom: 4,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 8,
    borderRadius: 4,
    fontSize: 16,
  },
});

export default EditDataScreen;
