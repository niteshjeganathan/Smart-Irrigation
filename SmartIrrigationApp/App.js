import React, {useEffect} from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import HomeScreen from './screens/HomeScreen';
import FarmDetailScreen from './screens/FarmDetailScreen';
import AddFarm from './screens/AddFarm';
import AsyncStorage from '@react-native-async-storage/async-storage';
import EditDataScreen from './screens/EditDataScreen';



const Stack = createStackNavigator();

const clearAsyncStorage = async () => {
  try {
    await AsyncStorage.clear();
    console.log('AsyncStorage cleared successfully!');
  } catch (error) {
    console.error('Error clearing AsyncStorage:', error);
  }
};

export default function App() {

  useEffect(() => {
    // Call clearAsyncStorage only once when the app starts
    //clearAsyncStorage();
  }, []);
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={HomeScreen} options={{ title: 'Home' }} />
        <Stack.Screen name="FarmDetail" component={FarmDetailScreen} options={{ title: 'Farm Details' }} />
        <Stack.Screen name="AddFarm" component={AddFarm} options={{ title: 'Add Farm' }}/>
        <Stack.Screen name="EditData" component={EditDataScreen} options={{ title: 'Edit Data' }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
