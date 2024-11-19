import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Alert, Keyboard, ScrollView, TouchableWithoutFeedback, TouchableOpacity, ActivityIndicator } from 'react-native';
import { getDatabase, ref, get, set } from 'firebase/database';

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
  const [isEditMode, setIsEditMode] = useState(false);
  const [prediction, setPrediction] = useState(null); // State for storing the prediction result
  const [loading, setLoading] = useState(true); // State for showing a loading spinner
  const month = parseInt(selectedDate.split('-')[1]); // Extract month from the date (assuming YYYY-MM-DD format)

  const labels = {
    T_max: 'Max Temperature',
    T_min: 'Min Temperature',
    Humidity_1: 'Max Humidity',
    Humidity_2: 'Min Humidity',
    WindSpeed: 'Wind Speed',
    Sunshine_Hours: 'Sunshine Hours',
  };

  const order = ['T_max', 'T_min', 'Humidity_1', 'Humidity_2', 'WindSpeed', 'Sunshine_Hours']; // Order of keys

  useEffect(() => {
    const fetchDataAndPredict = async () => {
      try {
        setLoading(true); // Show loading spinner
        const database = getDatabase();
        const dataRef = ref(database, `farms/${farmName}/dates/${selectedDate}`);
        const snapshot = await get(dataRef);

        if (snapshot.exists()) {
          const data = snapshot.val();
          setWeatherData(data);

          // Check if all values are present and run prediction
          const areAllValuesFilled = Object.values(data).every((value) => value !== '');
          if (areAllValuesFilled) {
            await runInference(data);
          }
        }
      } catch (error) {
        console.error("Error fetching data:", error);
        Alert.alert("Error", "Failed to fetch weather data.");
      } finally {
        setLoading(false); // Hide loading spinner
      }
    };

    fetchDataAndPredict();
  }, [farmName, selectedDate]);

  const toggleEditMode = () => {
    setIsEditMode((prevMode) => !prevMode);
  };

  useEffect(() => {
    navigation.setOptions({
      title: `Data for ${selectedDate}`,
      headerRight: () => (
        <TouchableOpacity onPress={toggleEditMode} style={styles.headerButton}>
          <Text style={styles.headerButtonText}>{isEditMode ? 'Cancel' : 'Edit'}</Text>
        </TouchableOpacity>
      ),
    });
  }, [navigation, isEditMode]);

  const runInference = async (data) => {
    const apiUrl = "http://172.20.10.3:5000/predict"; // Replace with your deployed API URL

    const inputData = [
      month,
      parseFloat(data.T_max),
      parseFloat(data.T_min),
      parseFloat(data.Humidity_1),
      parseFloat(data.Humidity_2),
      parseFloat(data.WindSpeed),
      parseFloat(data.Sunshine_Hours),
    ];

    console.log("Input Data for Prediction:", inputData);

    try {
      setLoading(true); // Show loading spinner during prediction
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ inputs: inputData }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result.prediction); // Update the prediction state
    } catch (error) {
      console.error("Error during API call:", error);
      Alert.alert("Error", "Failed to fetch prediction.");
    } finally {
      setLoading(false); // Hide loading spinner
    }
  };

  const handleChange = (key, value) => {
    setWeatherData((prevData) => ({ ...prevData, [key]: value }));
  };

  const handleSave = async () => {
    try {
      setLoading(true); // Show loading spinner
      const database = getDatabase();
  
      // Path to the specific farm and date
      const dataRef = ref(database, `farms/${farmName}/dates/${selectedDate}`);
  
      // Update the weather data for the specific date
      await set(dataRef, weatherData);
  
      Alert.alert("Success", "Data saved successfully!");
      setIsEditMode(false); // Exit edit mode
      // Reload the component
      navigation.replace("EditDataScreen", { farmName, selectedDate });
    } catch (error) {
      console.error("Error saving data:", error);
      Alert.alert("Error", "Failed to save data.");
    } finally {
      setLoading(false); // Hide loading spinner
    }
  };

  return (
    <TouchableWithoutFeedback onPress={() => Keyboard.dismiss()}>
      <View style={{ flex: 1 }}>
        {loading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator size="large" color="#007BFF" />
            <Text style={styles.loadingText}>Loading...</Text>
          </View>
        )}

        <ScrollView
          contentContainerStyle={styles.container}
          keyboardShouldPersistTaps="handled"
        >
          <View style={styles.nonEditableFieldContainer}>
            <Text style={styles.label}>Month:</Text>
            <Text style={styles.value}>{month}</Text>
          </View>

          {order.map((key) => (
            <View
              key={key}
              style={[
                styles.fieldContainer,
                !weatherData[key] && !isEditMode && styles.redContainer,
              ]}
            >
              <Text style={styles.label}>{labels[key]}:</Text>
              {isEditMode ? (
                <TextInput
                  style={styles.input}
                  value={String(weatherData[key])}
                  onChangeText={(value) => handleChange(key, value)}
                  keyboardType="numeric"
                />
              ) : (
                <Text
                  style={[
                    styles.value,
                    !weatherData[key] && styles.emptyValue,
                  ]}
                >
                  {String(weatherData[key]) || "Empty"}
                </Text>
              )}
            </View>
          ))}

          {!isEditMode && prediction !== null && (
            <View style={styles.predictionContainer}>
              <Text style={styles.predictionText}>
                Predicted Value: {parseFloat(prediction).toFixed(2)}
              </Text>
            </View>
          )}
        </ScrollView>

        {isEditMode && (
          <View style={styles.saveButtonContainer}>
            <Button title="Save" onPress={handleSave} color="#007BFF" />
          </View>
        )}
      </View>
    </TouchableWithoutFeedback>
  );
};

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 16,
    backgroundColor: "#fff",
  },
  nonEditableFieldContainer: {
    backgroundColor: "#e9e9e9",
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#ccc",
  },
  fieldContainer: {
    backgroundColor: "rgba(144, 238, 144, 0.3)", // Light green with opacity
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(144, 238, 144, 0.7)",
  },
  redContainer: {
    backgroundColor: "rgba(255, 0, 0, 0.2)",
    borderColor: "rgba(255, 0, 0, 0.7)",
  },
  label: {
    fontSize: 16,
    flex: 1,
  },
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    padding: 8,
    borderRadius: 4,
    fontSize: 16,
    flex: 1,
  },
  value: {
    fontSize: 16,
    color: "#333",
    flex: 1,
    textAlign: "right",
  },
  emptyValue: {
    color: "red",
  },
  headerButton: {
    marginRight: 16,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: "#007BFF",
    borderRadius: 4,
  },
  headerButtonText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "bold",
  },
  predictionContainer: {
    marginTop: 16,
    padding: 12,
    backgroundColor: "rgba(0, 128, 255, 0.1)",
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(0, 128, 255, 0.5)",
  },
  predictionText: {
    fontSize: 16,
    color: "#007BFF",
  },
  loadingOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 10,
  },
  loadingText: {
    marginTop: 10,
    color: "#fff",
    fontSize: 16,
  },
  saveButtonContainer: {
    padding: 16,
    backgroundColor: "#fff",
  },
});

export default EditDataScreen;
