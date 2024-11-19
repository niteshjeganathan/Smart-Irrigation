import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import { Calendar } from 'react-native-calendars';
import moment from 'moment';
import { getFarmDetails } from '../util/firebase';

const FarmDetailScreen = ({ route, navigation }) => {
  const { farmName } = route.params;

  const [farmData, setFarmData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFarmData = async () => {
      try {
        const data = await getFarmDetails(farmName);
        if (data) {
          setFarmData({ ...data, farmName }); // Ensure farmName is part of farmData
        } else {
          Alert.alert("Error", "Farm details not found.");
        }
      } catch (error) {
        console.error("Error fetching farm details:", error);
        Alert.alert("Error", "Could not fetch farm details.");
      } finally {
        setLoading(false);
      }
    };

    fetchFarmData();
  }, [farmName]);

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#000" />
      </View>
    );
  }

  if (!farmData) {
    return (
      <View style={styles.centered}>
        <Text>No farm details available</Text>
      </View>
    );
  }

  const today = moment().format('YYYY-MM-DD');
  const { cropType, sowDate, harvestDate } = farmData;

  const formattedSowDate = moment(sowDate).format('DD-MM-YYYY');
  const formattedHarvestDate = moment(harvestDate).format('DD-MM-YYYY');

  const generateMarkedDates = (sowDate, today) => {
    let markedDates = {};

    const startDate = moment(sowDate);
    let currentDate = startDate;

    while (currentDate.isBefore(today)) {
      const dateStr = currentDate.format('YYYY-MM-DD');
      markedDates[dateStr] = {
        selected: true,
        selectedColor: 'green',
        textColor: 'black',
      };
      currentDate = currentDate.add(1, 'days');
    }

    markedDates[today] = {
      selected: true,
      selectedColor: 'orange',
      customStyles: {
        text: { color: 'black', fontWeight: 'bold' },
      },
    };

    return markedDates;
  };

  const markedDates = generateMarkedDates(sowDate, today);

  return (
    <View style={styles.container}>
      <View style={styles.tile}>
        <Text style={styles.farmName}>{farmData.farmName}</Text>
        <Text style={styles.detail}>Crop Type: {cropType}</Text>
        <Text style={styles.detail}>Sow Date: {formattedSowDate}</Text>
        <Text style={styles.detail}>Harvest Date: {formattedHarvestDate}</Text>
      </View>
      <Calendar
        current={sowDate}
        minDate={sowDate}
        maxDate={harvestDate}
        markedDates={markedDates}
        onDayPress={(day) => {
          navigation.navigate('EditData', {
            farmName,
            selectedDate: day.dateString,
          });
          console.log(day.dateString)
        }}
        hideExtraDays={true}
        theme={{
          todayTextColor: 'black',
          dayTextColor: 'black',
          monthTextColor: 'black',
          textSectionTitleColor: 'black',
          arrowColor: 'green',
        }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#fff',
  },
  tile: {
    backgroundColor: '#f0f0f0',
    padding: 16,
    borderRadius: 10,
    marginBottom: 16,
  },
  farmName: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  detail: {
    fontSize: 18,
    marginBottom: 4,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default FarmDetailScreen;
