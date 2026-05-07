
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level3/main - 001.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'bb8a34UmdRErJ42JemF+/mT', 'main - 001');
// 3.16小游戏/command_TypeScript/level3/main - 001.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var gloabl_1 = require("./gloabl");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var mainCharacter = /** @class */ (function (_super) {
    __extends(mainCharacter, _super);
    function mainCharacter() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    mainCharacter.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    mainCharacter.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (gloabl_1.default.attack == true) {
            gloabl_1.default.minion1_hp -= 30;
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        gloabl_1.default.attack = false;
    };
    mainCharacter.prototype.onCollisionStay = function (other) {
        gloabl_1.default.attack = false;
        gloabl_1.default.minion_attack = false;
    };
    mainCharacter.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        if (gloabl_1.default.minion1_hp <= 0) {
            other.node.active = false;
            gloabl_1.default.attack = false;
            gloabl_1.default.minion_attack = false;
            gloabl_1.default.main_hp = 163;
            gloabl_1.default.minion1_hp = 163;
            gloabl_1.default.minion2_hp = 163;
            gloabl_1.default.minion3_hp = 163;
            cc.director.loadScene("fight4");
        }
        else {
            other.node.children[0].setContentSize(gloabl_1.default.minion1_hp, 19);
        }
    };
    mainCharacter.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    mainCharacter.prototype.update = function (dt) {
        if (gloabl_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], mainCharacter.prototype, "label", void 0);
    __decorate([
        property
    ], mainCharacter.prototype, "text", void 0);
    mainCharacter = __decorate([
        ccclass
    ], mainCharacter);
    return mainCharacter;
}(cc.Component));
exports.default = mainCharacter;

cc._RF.pop();
                    }
                    if (nodeEnv) {
                        __define(__module.exports, __require, __module);
                    }
                    else {
                        __quick_compile_project__.registerModuleFunc(__filename, function () {
                            __define(__module.exports, __require, __module);
                        });
                    }
                })();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDNcXG1haW4gLSAwMDEudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7QUFDbEYsbUNBQTZCO0FBS3ZCLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQTJDLGlDQUFZO0lBQXZEO1FBQUEscUVBbUZDO1FBaEZHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7SUE2RTNCLENBQUM7SUF2RUcsd0JBQXdCO0lBRXhCLDhCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDM0IsQ0FBQztJQUdELFNBQVM7SUFDVCx3Q0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFDdkIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxNQUFNLEdBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBRXpCLElBQUcsZ0JBQU0sQ0FBQyxNQUFNLElBQUksSUFBSSxFQUFDO1lBQ3JCLGdCQUFNLENBQUMsVUFBVSxJQUFJLEVBQUUsQ0FBQztZQUN4QixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUM3QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDckQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztRQUVELGdCQUFNLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztJQUMxQixDQUFDO0lBR0QsdUNBQWUsR0FBZixVQUFnQixLQUFLO1FBQ2pCLGdCQUFNLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztRQUN0QixnQkFBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7SUFDakMsQ0FBQztJQUVELHVDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUNsQixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBS2YsSUFBRyxnQkFBTSxDQUFDLFVBQVUsSUFBSSxDQUFDLEVBQUU7WUFDMUIsS0FBSyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzFCLGdCQUFNLENBQUMsTUFBTSxHQUFFLEtBQUssQ0FBQztZQUNyQixnQkFBTSxDQUFDLGFBQWEsR0FBRyxLQUFLLENBQUM7WUFDN0IsZ0JBQU0sQ0FBQyxPQUFPLEdBQUUsR0FBRyxDQUFDO1lBQ3BCLGdCQUFNLENBQUMsVUFBVSxHQUFFLEdBQUcsQ0FBQztZQUN2QixnQkFBTSxDQUFDLFVBQVUsR0FBRSxHQUFHLENBQUM7WUFDdkIsZ0JBQU0sQ0FBQyxVQUFVLEdBQUcsR0FBRyxDQUFDO1lBQ3hCLEVBQUUsQ0FBQyxRQUFRLENBQUMsU0FBUyxDQUFDLFFBQVEsQ0FBQyxDQUFDO1NBQ2hDO2FBQU07WUFDTixLQUFLLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxjQUFjLENBQUMsZ0JBQU0sQ0FBQyxVQUFVLEVBQUUsRUFBRSxDQUFDLENBQUM7U0FDN0Q7SUFFSixDQUFDO0lBRUQsNkJBQUssR0FBTDtRQUVBLElBQUksQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO0lBR25DLENBQUM7SUFFRCw4QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUNOLElBQUksZ0JBQU0sQ0FBQyxNQUFNLElBQUssSUFBSSxFQUFFO1lBQzNCLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBRzVFO2FBQU07WUFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQyxFQUFJO2dCQUNqRyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMzRTtTQUVKO0lBRUwsQ0FBQztJQTlFRDtRQURDLFFBQVEsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDO2dEQUNJO0lBR3ZCO1FBREMsUUFBUTsrQ0FDYztJQU5OLGFBQWE7UUFEakMsT0FBTztPQUNhLGFBQWEsQ0FtRmpDO0lBQUQsb0JBQUM7Q0FuRkQsQUFtRkMsQ0FuRjBDLEVBQUUsQ0FBQyxTQUFTLEdBbUZ0RDtrQkFuRm9CLGFBQWEiLCJmaWxlIjoiIiwic291cmNlUm9vdCI6Ii8iLCJzb3VyY2VzQ29udGVudCI6WyIvLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcbmltcG9ydCBnbG9hYmwgZnJvbSBcIi4vZ2xvYWJsXCJcclxuXHJcblxyXG5cclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgbWFpbkNoYXJhY3RlciBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcblxyXG4gICAgQHByb3BlcnR5KGNjLkxhYmVsKVxyXG4gICAgbGFiZWw6IGNjLkxhYmVsID0gbnVsbDtcclxuXHJcbiAgICBAcHJvcGVydHlcclxuICAgIHRleHQ6IHN0cmluZyA9ICdoZWxsbyc7XHJcbiAgICBtYWluX3g6IG51bWJlcjtcclxuICAgIFxyXG4gICAgcHVibGljIHN0YXRpYyBjdXJyZW50X3g6IG51bWJlcjtcclxuICAgIHB1YmxpYyBzdGF0aWMgY3VycmVudF95OiBudW1iZXI7XHJcblxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcblxyXG4gICAgb25Mb2FkICgpIHtcclxuICAgICAgICB2YXIgbWFuYWdlciA9IGNjLmRpcmVjdG9yLmdldENvbGxpc2lvbk1hbmFnZXIoKTtcclxuICAgICAgICBtYW5hZ2VyLmVuYWJsZWQgPSB0cnVlO1xyXG4gICAgfVxyXG4gICAgXHJcblxyXG4gICAgLy/kuqfnlJ/norDmkp7kvJrosIPnlKhcclxuICAgIG9uQ29sbGlzaW9uRW50ZXIob3RoZXIsc2VsZil7XHJcbiAgICAgICAgY2MubG9nKFwi5byA5aeL56Kw5pKeXCIrb3RoZXIudGFnKTtcclxuICAgICAgIFxyXG4gICAgICAgIGlmKGdsb2FibC5hdHRhY2sgPT0gdHJ1ZSl7XHJcbiAgICAgICAgICAgIGdsb2FibC5taW5pb24xX2hwIC09IDMwO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL3h5L21haW5fZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIi0zMFwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZTIuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIlwiO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBnbG9hYmwuYXR0YWNrID0gZmFsc2U7XHJcbiAgICB9XHJcblxyXG5cclxuICAgIG9uQ29sbGlzaW9uU3RheShvdGhlcikge1xyXG4gICAgICAgIGdsb2FibC5hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBnbG9hYmwubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgICAgY2MubG9nKFwi56Kw5pKe57uT5p2fXCIpO1xyXG4gICAgICAgXHJcbiAgICAgICBcclxuICAgICBcclxuICAgICAgXHJcbiAgICAgICBpZihnbG9hYmwubWluaW9uMV9ocCA8PSAwKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBnbG9hYmwuYXR0YWNrPSBmYWxzZTtcclxuICAgICAgICBnbG9hYmwubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgICAgIGdsb2FibC5tYWluX2hwPSAxNjM7XHJcbiAgICAgICAgZ2xvYWJsLm1pbmlvbjFfaHA9IDE2MztcclxuICAgICAgICBnbG9hYmwubWluaW9uMl9ocD0gMTYzO1xyXG4gICAgICAgIGdsb2FibC5taW5pb24zX2hwID0gMTYzO1xyXG4gICAgICAgIGNjLmRpcmVjdG9yLmxvYWRTY2VuZShcImZpZ2h0NFwiKTtcclxuICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZShnbG9hYmwubWluaW9uMV9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcbiAgICBcclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgXHJcbiAgICB0aGlzLm1haW5feCA9IHRoaXMubm9kZS5wb3NpdGlvbi54O1xyXG4gICAgICAgXHJcbiAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgdXBkYXRlIChkdCkge1xyXG4gICAgICAgIGlmIChnbG9hYmwuYXR0YWNrID09ICB0cnVlKSB7XHJcbiAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCArIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgXHJcbiAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgaWYgKCEodGhpcy5ub2RlLnBvc2l0aW9uLnggPD0gdGhpcy5tYWluX3ggKyA1MCkgJiYgKHRoaXMubm9kZS5wb3NpdGlvbi54ID49IHRoaXMubWFpbl94IC0gNTApKSAgIHtcclxuICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54IC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgIH1cclxuXHJcbiAgICB9XHJcbiAgICBcclxufVxyXG5cclxuXHJcbiJdfQ==