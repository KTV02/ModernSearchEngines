<template>
  <div class="search-component">
    <div class="logo-box">
      <img src="@/assets/logoUni.png" alt="Logo" class="logo" />
    </div>
    <div class="title-box">
      <h1>Search Engine</h1>
    </div>
    <form @submit.prevent="submitQuery" class="search-form">
      <input v-model="query" type="text" placeholder="Enter your search query" class="search-input" />
      <button type="submit" class="search-button">Search</button>
      <button type="button" class="tree-button" @click="toggleTree">Display Topic modeling</button>
    </form>
    <div v-if="results.relevantTitles && results.relevantTitles.length" class="results">
      <div v-for="(title, index) in results.relevantTitles" :key="index" class="result-item">
        <a :href="results.relevantUrls[index]" target="_blank" class="result-link">{{ title }}</a>
      </div>
    </div>
    <Modal_component v-if="showTree" @close="toggleTree">
      <LinkedBoxes />
    </Modal_component>
  </div>
</template>


<script>
import axios from 'axios';
import Modal_component from './Modal_component.vue';
import LinkedBoxes from './LinkedBoxes.vue';

export default {
  components: {
    LinkedBoxes,
    Modal_component
  },
  data() {
    return {
      query: '',
      results: {
        relevantTitles: [],
        relevantUrls: []
      },
      showTree: false
    };
  },
  methods: {
    async submitQuery() {
      try {
        const response = await axios.post('http://localhost:5000/rank', {
          query: this.query
        });
        this.results = response.data;
      } catch (error) {
        console.error('Error fetching search results:', error);
      }
    },
    toggleTree() {
      this.showTree = !this.showTree;
    }
  }
};
</script>
<style scoped>
.search-component {
  max-width: 600px;
  margin: 0 auto;
  text-align: center;
}

.logo-box {
  background-color: white;
  padding: 20px;
  margin-bottom: 20px;
  border-radius: 12px; /* Add rounded corners */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.logo {
  width: 150px;
}

.title-box {
  background-color: white;
  padding: 10px;
  margin: 0 auto 20px auto;
  border-radius: 12px; /* Add rounded corners */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  display: inline-block; /* Make the box only as big as the text */
}

.title-box h1 {
  margin: 0;
}

.search-form {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 20px;
}

.search-input {
  width: 60%;
  padding: 10px;
  margin-right: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.search-button,
.tree-button {
  padding: 10px 20px;
  border: none;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap; /* Prevent text wrap */
}

.search-button {
  background-color: #007bff;
}

.search-button:hover {
  background-color: #0056b3;
}

.tree-button {
  background-color: #28a745;
  margin-left: 10px;
}

.tree-button:hover {
  background-color: #218838;
}

.results {
  text-align: left;
  margin-top: 20px;
  background-color: white; /* Set background to white */
  padding: 20px; /* Add padding */
  border-radius: 12px; /* Add rounded corners */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Add box shadow */
}

.result-item {
  margin-bottom: 10px;
}

.result-link {
  text-decoration: none;
  color: #007bff;
}

.result-link:hover {
  text-decoration: underline;
}
</style>
