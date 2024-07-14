<template>
  <div class="search-component">
    <h1>Search Engine</h1>
    <form @submit.prevent="submitQuery">
      <input v-model="query" type="text" placeholder="Enter your search query" />
      <button type="submit">Search</button>
    </form>
    <div v-if="results.length">
      <h2>Results:</h2>
      <ul>
        <li v-for="(result, index) in results" :key="index">{{ result }}</li>
      </ul>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      query: '',
      results: []
    };
  },
  methods: {
    async submitQuery() {
      try {
        const response = await axios.post('http://localhost:5000/rank', {
          query: this.query
        });
        this.results = response.data.results;
      } catch (error) {
        console.error('Error fetching search results:', error);
      }
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

input {
  width: 70%;
  padding: 8px;
  margin-right: 10px;
}

button {
  padding: 8px 16px;
}
</style>