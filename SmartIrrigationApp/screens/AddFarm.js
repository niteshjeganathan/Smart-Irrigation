import React, { useState } from 'react';
import { View, Text, Button, StyleSheet, TextInput, ActivityIndicator, Modal, Alert } from 'react-native';
import { getDatabase, ref, set } from 'firebase/database';
import DateTimePickerModal from 'react-native-modal-datetime-picker';

const AddFarm = ({ navigation }) => {
  const [farmName, setFarmName] = useState('');
  const [cropType, setCropType] = useState('');
  const [sowDate, setSowDate] = useState(null);
  const [harvestDate, setHarvestDate] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isDatePickerVisible, setDatePickerVisibility] = useState(false);
  const [activeDateField, setActiveDateField] = useState(null);

  const handleSave = async () => {
    if (!farmName) {
      Alert.alert("Error", "Please enter a unique farm name.");
      return;
    }

    setLoading(true);

    const newFarmData = { 
      cropType,
      sowDate: sowDate ? sowDate.toISOString() : null,
      harvestDate: harvestDate ? harvestDate.toISOString() : null,
    };

    try {
      const database = getDatabase();
      const farmRef = ref(database, `farms/${farmName}`); // Use farmName as the key

      await set(farmRef, newFarmData);

      Alert.alert("Success", "Farm created successfully.");
      navigation.goBack();
    } catch (error) {
      console.error("Error saving farm data:", error);
      Alert.alert("Error", "Failed to save data. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const showDatePicker = (field) => {
    setActiveDateField(field);
    setDatePickerVisibility(true);
  };

  const hideDatePicker = () => {
    setDatePickerVisibility(false);
  };

  const handleConfirm = (selectedDate) => {
    if (activeDateField === 'sowDate') {
      setSowDate(selectedDate);
    } else if (activeDateField === 'harvestDate') {
      setHarvestDate(selectedDate);
    }
    hideDatePicker();
  };

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Farm Name:</Text>
      <TextInput
        style={styles.input}
        placeholder="Enter unique farm name"
        value={farmName}
        onChangeText={setFarmName}
      />

      <Text style={styles.label}>Crop Type:</Text>
      <TextInput
        style={styles.input}
        placeholder="Enter crop type"
        value={cropType}
        onChangeText={setCropType}
      />

      <Text style={styles.label}>Sow Date:</Text>
      <Button title="Select Sow Date" onPress={() => showDatePicker('sowDate')} />
      <Text style={styles.dateText}>{sowDate ? sowDate.toDateString() : 'No date selected'}</Text>

      <Text style={styles.label}>Harvest Date:</Text>
      <Button title="Select Harvest Date" onPress={() => showDatePicker('harvestDate')} />
      <Text style={styles.dateText}>{harvestDate ? harvestDate.toDateString() : 'No date selected'}</Text>

      <DateTimePickerModal
        isVisible={isDatePickerVisible}
        mode="date"
        onConfirm={handleConfirm}
        onCancel={hideDatePicker}
      />

      <Button title="Save Farm" onPress={handleSave} />

      {loading && (
        <Modal transparent={true} animationType="none" visible={loading}>
          <View style={styles.overlay}>
            <ActivityIndicator size="large" color="#fff" />
          </View>
        </Modal>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#fff',
  },
  label: {
    fontSize: 18,
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 8,
    marginBottom: 16,
    borderRadius: 4,
  },
  dateText: {
    fontSize: 16,
    marginVertical: 8,
    color: '#555',
  },
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
});

export default AddFarm;
