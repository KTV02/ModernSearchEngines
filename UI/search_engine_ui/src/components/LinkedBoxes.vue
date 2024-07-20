<template>
  <div class="linked-boxes">
    <div class="center-dot"></div>
    <div v-for="(box, index) in boxes" :key="index" :class="'box box' + (index + 1)" :style="{ backgroundColor: box.color }">
      <div class="box-name">{{ box.name }}</div>
      <div class="links">
        <div v-for="(link, linkIndex) in box.urls" :key="linkIndex" class="link">
          <a :href="link.url" target="_blank">{{ link.title }}</a>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'LinkedBoxes',
  data() {
    return {
      boxes: []
    };
  },
  created() {
    this.fetchLinks();
  },
  methods: {
    async fetchLinks() {
      try {
        const response = await axios.get('http://localhost:5000/get_links');
        this.boxes = response.data;
      } catch (error) {
        console.error('Error fetching links:', error);
      }
    }
  }
};
</script>

<style scoped>
.linked-boxes {
  position: relative;
  width: 600px;
  height: 600px;
  margin: 0 auto;
  display: flex;
  justify-content: center;
  align-items: center;
}

.center-dot {
  width: 10px;
  height: 10px;
  background-color: black;
  border-radius: 50%;
  position: absolute;
}

.box {
  position: absolute;
  min-width: 100px;
  max-width: 230px; /* Allow boxes to grow but not exceed a certain width */
  padding: 10px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  border: 1px solid #ccc;
  border-radius: 8px;
  background-color: #f9f9f9;
  box-sizing: border-box; /* Ensure padding and border are included in the element's total width and height */
  transform-origin: center;
}

.box-name {
  font-weight: bold;
  margin-bottom: 10px;
}

.links {
  display: flex;
  flex-direction: column;
}

.link {
  margin: 5px 0;
  word-wrap: break-word; /* Break long words to fit within the box */
}


.box1 {
  transform: rotate(270deg) translate(250px) rotate(-270deg);
}

.box2 {
  transform: rotate(198deg) translate(250px) rotate(-198deg);
}

.box3 {
  transform: rotate(342deg) translate(250px) rotate(-342deg);
}

.box4 {
  transform: rotate(126deg) translate(250px) rotate(-126deg);
}

.box5 {
  transform: rotate(54deg) translate(250px) rotate(-54deg);
}
</style>