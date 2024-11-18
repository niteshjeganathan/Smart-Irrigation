import React, { useState, useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions, FlatList, ActivityIndicator } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { getFarmDetails } from '../util/firebase'; // Import the function to get all farms

const HomeScreen = ({ navigation }) => {
  const [farms, setFarms] = useState([]);
  const [loading, setLoading] = useState(true);

  // Function to load all farms from Firebase using getFarmDetails
  const loadFarms = async () => {
    setLoading(true);
    try {
      const allFarms = await getFarmDetails(); // Use the function to get all farms
      if (allFarms) {
        const farmsData = Object.keys(allFarms).map((key) => ({
          ...allFarms[key],
          farmName: key,
        }));
        setFarms(farmsData);
      } else {
        console.warn("No farms found in the database.");
        setFarms([]);
      }
    } catch (error) {
      console.error('Error loading farms:', error);
    } finally {
      setLoading(false);
    }
  };

  // Reload farms when screen is focused
  useFocusEffect(
    useCallback(() => {
      loadFarms();
    }, [])
  );

  const renderFarm = ({ item }) => (
    <TouchableOpacity
      style={styles.button}
      onPress={() =>
        navigation.navigate('FarmDetail', {
          farmName: item.farmName,
          cropType: item.cropType,
        })
      }
    >
      <Text style={styles.buttonText}>{item.farmName}</Text>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.addButton}
        onPress={() => navigation.navigate('AddFarm')}
      >
        <Text style={styles.addButtonText}>Add Farm</Text>
      </TouchableOpacity>
      {loading ? (
        <ActivityIndicator size="large" color="#4CAF50" style={styles.loader} />
      ) : (
        <FlatList
          data={farms}
          renderItem={renderFarm}
          keyExtractor={(item) => item.farmName}
          numColumns={2}
          columnWrapperStyle={styles.columnWrapper}
          contentContainerStyle={styles.scrollContainer}
        />
      )}
    </View>
  );
};

const { width } = Dimensions.get('window');
const buttonSize = width * 0.45;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#fff',
  },
  addButton: {
    backgroundColor: '#4CAF50',
    padding: 10,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 10,
  },
  addButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  loader: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scrollContainer: {
    paddingBottom: 16,
  },
  columnWrapper: {
    justifyContent: 'space-between',
  },
  button: {
    width: buttonSize,
    height: buttonSize,
    marginVertical: 8,
    backgroundColor: '#4CAF50',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 10,
  },
  buttonText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
  },
});

export default HomeScreen;
